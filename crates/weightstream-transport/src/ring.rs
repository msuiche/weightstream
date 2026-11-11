//! The 4-rank ring: one RC QP per (neighbor, rail), a registered host
//! buffer with per-source-rank receive slots, and the one-shot two-hop
//! fp32 all-reduce from docs/transport.md.
//!
//! All-reduce shape (n = 4):
//!   hop 1: rank r writes its partial to BOTH neighbors (rail 0 toward the
//!          successor, rail 1 toward the predecessor), imm = source rank
//!   hop 2: rank r forwards the partial it got from its predecessor to
//!          its successor
//!   Every rank then holds all 4 partials after exactly two hops and sums
//!   them locally in rank order 0+1+2+3 (fixed order -> batch-independent,
//!   bit-identical numerics). Traffic per rank: 3 sends, 3 receives.
//!
//! Rail usage per step: every successor-bound message crosses rail 0 of
//! that cable, every predecessor-bound message rail 1 — the two rails of
//! a link each carry one direction, no lane changes needed mid-collective.
//!
//! Run-ahead safety: immediate data carries (step seq:16 | src rank:8).
//! Slots are double-banked by step parity; a rank can be at most one step
//! ahead of a neighbor (progress requires the neighbor's next partial), so
//! two banks plus per-step masks make early completions unambiguous
//! without a per-step barrier.
//!
//! Fail-stop: every wait has a deadline (default 100 ms). A timeout or any
//! CQE error aborts the operation — the engine must never compute on
//! partially written buffers.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::nodes::RingCfg;
use crate::verbs::{PollBatch, PollEvent, PollKind, PortRef, Qp, Verbs, VerbsError, VerbsResult};

const RECV_BACKLOG: usize = 128;
// ping tag: the low byte of an all-reduce imm is a source rank (0..3), so
// 0xff can never collide with a collective immediate
const PING_IMM: u32 = 0xff;

pub struct RailEnd {
    pub qpn: u32,
    pub gid: [u8; 16],
    pub rkey: u32,
    pub slots_addr: u64,
}

impl Clone for RailEnd {
    fn clone(&self) -> Self {
        RailEnd {
            qpn: self.qpn,
            gid: self.gid,
            rkey: self.rkey,
            slots_addr: self.slots_addr,
        }
    }
}

pub struct LinkMeta {
    /// ibverbs device name (the node map's interface entry)
    pub ibdev: String,
    pub port: u8,
}

pub struct Ring {
    pub nranks: usize,
    pub me: u8,
    pub elems: usize,
    verbs: Verbs,
    /// the registered slot buffer; keeps the allocation pinned while its
    /// regions are registered with the HCAs
    #[allow(dead_code)]
    buf: Box<[u8]>,
    base: u64,
    /// device -> MR over `buf` (the rkey a peer needs depends on which of
    /// my ports its write lands on)
    mrs: BTreeMap<usize, crate::verbs::MrInfo>,
    /// (peer, rail) -> device index
    qp_dev: BTreeMap<(u8, usize), usize>,
    /// (peer, rail) -> local iface/ibdev/port, for counter snapshots
    pub links: BTreeMap<(u8, usize), LinkMeta>,
    qps: BTreeMap<(u8, usize), Qp>,
    qpn_map: BTreeMap<u32, (u8, usize)>,
    /// peer -> per-rail remote endpoints, filled after ctrl exchange
    peers: BTreeMap<u8, [RailEnd; 2]>,
    recv_posted: BTreeMap<(u8, usize), usize>,
    pending_sends: BTreeMap<(u8, usize), usize>,
    step: u64,
    cur_seq: u16,
    cur_mask: u32,
    next_mask: u32,
    wait: Duration,
    out_off: u64,
    ping_off: u64,
    gather_off: u64,
    /// allreduce + allgather latency accounting (µs), for engine reports
    pub coll_us: f64,
    pub coll_count: u64,
}

fn wrap_verbs(e: VerbsError) -> String {
    format!("{e}")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn imm(seq: u16, src: u8) -> u32 {
    ((seq as u32) << 8) | src as u32
}

impl Ring {
    /// Discover the ports serving my four link interfaces, register the
    /// slot buffer on each HCA involved, create the four QPs (INIT).
    /// Returns (ring, endpoints JSON to advertise to the head).
    pub fn new(
        verbs: Verbs,
        cfg: &RingCfg,
        elems: usize,
        wait: Duration,
    ) -> VerbsResult<(Ring, serde_json::Value)> {
        assert_eq!(cfg.nranks, 4, "the two-hop all-reduce is a 4-rank design");
        let slot_bytes = elems * 4;
        let out_off = 8 * slot_bytes as u64;
        let ping_off = 9 * slot_bytes as u64;
        let gather_off = 9 * slot_bytes as u64 + 8;
        let mut buf = vec![0u8; 13 * slot_bytes + 8].into_boxed_slice();
        let base = buf.as_ptr() as u64;

        // resolve every (peer, rail) interface to its device/port/gid
        let mut port_refs: BTreeMap<(u8, usize), PortRef> = BTreeMap::new();
        let mut links = BTreeMap::new();
        let mut ips: BTreeMap<(u8, usize), String> = BTreeMap::new();
        let mut eps = Vec::new();
        for (peer, link) in [(cfg.succ.rank, &cfg.succ), (cfg.pred.rank, &cfg.pred)] {
            for rail in 0..2 {
                let iface = &link.rails[rail].local;
                let subnet = &link.rails[rail].subnet;
                // the map's interface names are ibverbs device names
                // (netdev names like enp1s0f0np0 may differ)
                let port_ref = verbs.find_port_in_subnet(iface, subnet, cfg.gid_index)?;
                links.insert(
                    (peer, rail),
                    LinkMeta {
                        ibdev: iface.clone(),
                        port: port_ref.port,
                    },
                );
                ips.insert((peer, rail), port_ref.ip.clone());
                port_refs.insert((peer, rail), port_ref);
            }
        }

        // register the slot buffer once per distinct HCA on the rank's links
        let mut mrs = BTreeMap::new();
        for key in port_refs.keys() {
            let dev = port_refs[key].dev;
            if !mrs.contains_key(&dev) {
                mrs.insert(dev, verbs.register(dev, &mut buf)?);
            }
        }

        // create QPs and advertise endpoints
        let mut qps = BTreeMap::new();
        let mut qpn_map = BTreeMap::new();
        let mut qp_dev = BTreeMap::new();
        let mut recv_posted = BTreeMap::new();
        let mut pending_sends = BTreeMap::new();
        for (key, port_ref) in &port_refs {
            let qp = verbs.create_qp(port_ref)?;
            let dev = port_ref.dev;
            qp_dev.insert(*key, dev);
            qpn_map.insert(qp.qpn, *key);
            recv_posted.insert(*key, 0);
            pending_sends.insert(*key, 0);
            eps.push(serde_json::json!({
                "to": key.0,
                "rail": key.1,
                "qpn": qp.qpn,
                "gid": hex(&port_ref.gid),
                "rkey": mrs[&dev].rkey,
                "slots": format!("{base:#x}"),
                "ip": ips[key],
            }));
            qps.insert(*key, qp);
        }

        let endpoints = serde_json::json!({ "rank": cfg.me, "eps": eps });
        Ok((
            Ring {
                nranks: 4,
                me: cfg.me,
                elems,
                verbs,
                buf,
                base,
                mrs,
                qp_dev,
                links,
                qps,
                qpn_map,
                peers: BTreeMap::new(),
                recv_posted,
                pending_sends,
                step: 0,
                cur_seq: u16::MAX, // first step uses seq 0
                cur_mask: 0,
                next_mask: 0,
                wait,
                out_off,
                ping_off,
                gather_off,
                coll_us: 0.0,
                coll_count: 0,
            },
            endpoints,
        ))
    }

    /// Apply the head's aggregate endpoint map and connect every QP (RTR).
    /// Only this rank's two ring neighbors have endpoints toward it; the
    /// others are not looked up.
    pub fn to_rtr(&mut self, all: &serde_json::Value) -> Result<(), String> {
        let entries = all
            .as_array()
            .ok_or("endpoint map must be an array of per-rank objects")?;
        let find_peer_ep = |peer: u8, rail: usize| -> Result<RailEnd, String> {
            let eps = entries
                .iter()
                .find(|ep| ep["rank"].as_u64() == Some(peer as u64))
                .ok_or_else(|| format!("endpoint map missing rank {peer}"))?["eps"]
                .as_array()
                .ok_or("eps missing")?;
            let e = eps
                .iter()
                .find(|e| {
                    e["to"].as_u64() == Some(self.me as u64)
                        && e["rail"].as_u64() == Some(rail as u64)
                })
                .ok_or_else(|| {
                    format!(
                        "rank {peer} has no endpoint toward rank {} rail {rail}",
                        self.me
                    )
                })?;
            let mut end = RailEnd {
                qpn: e["qpn"].as_u64().ok_or("qpn")? as u32,
                gid: [0u8; 16],
                rkey: e["rkey"].as_u64().ok_or("rkey")? as u32,
                slots_addr: 0,
            };
            let hexs = e["gid"].as_str().ok_or("gid must be hex")?;
            for i in 0..16 {
                end.gid[i] = u8::from_str_radix(&hexs[2 * i..2 * i + 2], 16)
                    .map_err(|_| "bad gid hex".to_string())?;
            }
            end.slots_addr = u64::from_str_radix(
                e["slots"]
                    .as_str()
                    .ok_or("slots addr")?
                    .trim_start_matches("0x"),
                16,
            )
            .map_err(|_| "bad slots addr")?;
            Ok(end)
        };
        for peer in self.qps.keys().map(|k| k.0).collect::<Vec<_>>() {
            let ends = [find_peer_ep(peer, 0)?, find_peer_ep(peer, 1)?];
            self.peers.insert(peer, ends);
        }
        for (key, qp) in &self.qps {
            let end = &self.peers[&key.0][key.1];
            qp.to_rtr(end.qpn, end.gid).map_err(wrap_verbs)?;
        }
        Ok(())
    }

    /// Second half of QP bring-up, after every rank reached RTR.
    pub fn to_rts(&mut self) -> Result<(), String> {
        for qp in self.qps.values() {
            qp.to_rts().map_err(wrap_verbs)?;
        }
        for key in self.qps.keys().cloned().collect::<Vec<_>>() {
            self.ensure_recv(key)?;
        }
        Ok(())
    }

    fn ensure_recv(&mut self, key: (u8, usize)) -> Result<(), String> {
        let posted = self.recv_posted[&key];
        if posted < RECV_BACKLOG {
            self.qps[&key]
                .post_recv(RECV_BACKLOG - posted)
                .map_err(wrap_verbs)?;
            self.recv_posted.insert(key, RECV_BACKLOG);
        }
        Ok(())
    }

    fn slot_addr(&self, bank: usize, rank: u8) -> u64 {
        self.base + ((bank * 4 + rank as usize) * self.elems * 4) as u64
    }

    fn write_slot(
        &self,
        key: (u8, usize),
        src_addr: u64,
        len: usize,
        bank: usize,
        src_rank: u8,
        seq: u16,
    ) -> Result<(), String> {
        let qp = &self.qps[&key];
        let end = &self.peers[&key.0][key.1];
        let lkey = self.mrs[&self.qp_dev[&key]].lkey;
        let remote = end.slots_addr + ((bank * 4 + src_rank as usize) * self.elems * 4) as u64;
        qp.write_imm(
            src_addr,
            (len * 4) as u32,
            lkey,
            remote,
            end.rkey,
            imm(seq, src_rank),
        )
        .map_err(wrap_verbs)
    }

    /// The two-hop exchange every collective shares: stage `input` into this
    /// step's bank slot for my rank, one-shot write it to both neighbors,
    /// forward the predecessor's partial to the successor, and wait until
    /// every rank's slot for this step has landed. `input.len()` may be
    /// smaller than the ring's slot capacity (`elems`) — the slot stride
    /// stays fixed, the wire payload shrinks. Returns (bank, len).
    ///
    /// The engine is SPMD-synchronous: every rank issues the same collective
    /// sequence, so a receiver always knows how to interpret the slots of the
    /// step it is currently in (all-reduce sums them, all-gather
    /// concatenates them) — the immediate carries only (seq, src).
    fn exchange(&mut self, input: &[f32]) -> Result<(usize, usize), String> {
        let len = input.len();
        if len == 0 || len > self.elems {
            return Err(format!(
                "collective input len {len} not in 1..={}",
                self.elems
            ));
        }
        let n = self.nranks;
        let me = self.me as usize;
        let succ = (me + 1) % n;
        let pred = (me + n - 1) % n;
        let fwd_src = (me + n - 2) % n; // partial forwarded to me by my predecessor
        let expected: u32 = (1 << succ) | (1 << pred) | (1 << fwd_src);

        self.step += 1;
        self.cur_seq = self.cur_seq.wrapping_add(1);
        let seq = self.cur_seq;
        let bank = (self.step & 1) as usize;
        self.cur_mask = self.next_mask;
        self.next_mask = 0;

        // stage my partial into this step's bank slot for my rank
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.slot_addr(bank, me as u8) as *mut f32,
                len,
            );
        }
        let my_slot = self.slot_addr(bank, me as u8);

        for key in [
            (succ as u8, 0),
            (succ as u8, 1),
            (pred as u8, 0),
            (pred as u8, 1),
        ] {
            self.ensure_recv(key)?;
        }

        // hop 1: my partial to both neighbors
        self.write_slot((succ as u8, 0), my_slot, len, bank, me as u8, seq)?;
        self.write_slot((pred as u8, 1), my_slot, len, bank, me as u8, seq)?;

        // hop 2: forward predecessor's partial to my successor once arrived
        self.wait_mask(1 << pred)?;
        let pred_slot = self.slot_addr(bank, pred as u8);
        self.write_slot((succ as u8, 0), pred_slot, len, bank, pred as u8, seq)?;

        self.wait_mask(expected)?;
        Ok((bank, len))
    }

    /// One-shot all-reduce of `input` (fp32, `input.len() <= elems`). On
    /// success returns the rank-ordered sum, in registered host memory,
    /// valid until the next collective on this ring.
    pub fn allreduce(&mut self, input: &[f32]) -> Result<&[f32], String> {
        let t0 = Instant::now();
        let (bank, len) = self.exchange(input)?;
        let n = self.nranks;
        // rank-ordered sum, left to right: slot[0]+slot[1]+slot[2]+slot[3]
        let out = unsafe {
            let out = (self.base + self.out_off) as *mut f32;
            for i in 0..len {
                let mut acc = *(self.slot_addr(bank, 0) as *const f32).add(i);
                for r in 1..n {
                    acc += *(self.slot_addr(bank, r as u8) as *const f32).add(i);
                }
                *out.add(i) = acc;
            }
            std::slice::from_raw_parts(out, len)
        };
        self.coll_us += t0.elapsed().as_secs_f64() * 1e6;
        self.coll_count += 1;
        Ok(out)
    }

    /// One-shot all-gather of `contribution` (fp32, `len <= elems`): every
    /// rank receives the rank-ordered concatenation
    /// [c_0 | c_1 | c_2 | c_3], in registered host memory, valid until the
    /// next collective. Same QPs, slots and two-hop schedule as the
    /// all-reduce — the receiver interprets the step's slots as
    /// contributions instead of partials.
    pub fn allgather(&mut self, contribution: &[f32]) -> Result<&[f32], String> {
        let t0 = Instant::now();
        let (bank, len) = self.exchange(contribution)?;
        let total = len * self.nranks;
        let out = unsafe {
            let out = (self.base + self.gather_off) as *mut f32;
            for r in 0..self.nranks {
                let src = self.slot_addr(bank, r as u8) as *const f32;
                std::ptr::copy_nonoverlapping(src, out.add(r * len), len);
            }
            std::slice::from_raw_parts(out, total)
        };
        self.coll_us += t0.elapsed().as_secs_f64() * 1e6;
        self.coll_count += 1;
        Ok(out)
    }

    fn wait_mask(&mut self, want: u32) -> Result<(), String> {
        let deadline = Instant::now() + self.wait;
        let mut batch = PollBatch::default();
        while self.cur_mask & want != want {
            self.poll_once(&mut batch)?;
            if Instant::now() >= deadline {
                for qp in self.qps.values() {
                    qp.to_err();
                }
                return Err(format!(
                    "all-reduce wait timeout after {} ms (mask {:#06x} of {:#06x}); fail-stop, aborting",
                    self.wait.as_millis(),
                    self.cur_mask,
                    want
                ));
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    fn poll_once(&mut self, batch: &mut PollBatch) -> Result<(), String> {
        let n = self.verbs.devices.len();
        for d in 0..n {
            let cq = self.verbs.devices[d].cq_ptr();
            let n = batch.poll(cq).map_err(wrap_verbs)?;
            for i in 0..n {
                self.handle_event(batch.event(i))?;
            }
        }
        Ok(())
    }

    fn handle_event(&mut self, ev: PollEvent) -> Result<(), String> {
        let key = *self
            .qpn_map
            .get(&ev.qp_num)
            .ok_or_else(|| format!("CQE from unknown qpn {}", ev.qp_num))?;
        match ev.kind {
            PollKind::Send => {
                let p = self.pending_sends.get_mut(&key).unwrap();
                *p = p.saturating_sub(1);
            }
            PollKind::Recv { imm: value } => {
                let posted = self.recv_posted.get_mut(&key).unwrap();
                *posted = posted.saturating_sub(1);
                if value == PING_IMM {
                    return Err("unexpected ping completion during all-reduce".into());
                }
                let seq = (value >> 8) as u16;
                let src = (value & 0xff) as u8;
                if seq == self.cur_seq {
                    self.cur_mask |= 1 << src;
                } else if seq == self.cur_seq.wrapping_add(1) {
                    self.next_mask |= 1 << src;
                } else {
                    return Err(format!(
                        "collective out of sequence: got step {seq} at step {} (fail-stop)",
                        self.cur_seq
                    ));
                }
            }
        }
        Ok(())
    }

    /// RDMA ping/pong over one (peer, rail): 8-byte write-with-imm each
    /// way, full round trips. The initiator returns RTT samples in µs; the
    /// responder returns empty.
    pub fn ping(
        &mut self,
        peer: u8,
        rail: usize,
        initiator: bool,
        iters: usize,
        warmup: usize,
    ) -> Result<Vec<f64>, String> {
        let key = (peer, rail);
        self.ensure_recv(key)?;
        let end = self.peers[&peer][rail].clone();
        let lkey = self.mrs[&self.qp_dev[&key]].lkey;
        let local_ping = self.base + self.ping_off;
        let remote_ping = end.slots_addr + self.ping_off;
        let mut rtts = Vec::with_capacity(iters);
        let mut batch = PollBatch::default();

        let wait_recv = |ring: &mut Ring,
                         batch: &mut PollBatch,
                         key: (u8, usize),
                         deadline: Instant|
         -> Result<(), String> {
            loop {
                let n = ring.verbs.devices.len();
                for d in 0..n {
                    let cq = ring.verbs.devices[d].cq_ptr();
                    let n = batch.poll(cq).map_err(wrap_verbs)?;
                    for i in 0..n {
                        let ev = batch.event(i);
                        let k = *ring.qpn_map.get(&ev.qp_num).ok_or("unknown qpn")?;
                        match ev.kind {
                            PollKind::Send => {
                                let p = ring.pending_sends.get_mut(&k).unwrap();
                                *p = p.saturating_sub(1);
                            }
                            PollKind::Recv { .. } => {
                                if k != key {
                                    return Err("unexpected recv on non-ping QP during ping".into());
                                }
                                let posted = ring.recv_posted.get_mut(&k).unwrap();
                                *posted = posted.saturating_sub(1);
                                return Ok(());
                            }
                        }
                    }
                }
                if Instant::now() >= deadline {
                    return Err("ping recv timeout (fail-stop)".into());
                }
                std::hint::spin_loop();
            }
        };

        if initiator {
            for i in 0..warmup + iters {
                let deadline = Instant::now() + Duration::from_secs(2);
                let t0 = Instant::now();
                self.qps[&key]
                    .write_imm(local_ping, 8, lkey, remote_ping, end.rkey, PING_IMM)
                    .map_err(wrap_verbs)?;
                wait_recv(self, &mut batch, key, deadline)?;
                if i >= warmup {
                    rtts.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                self.ensure_recv(key)?;
            }
        } else {
            for _ in 0..warmup + iters {
                let deadline = Instant::now() + Duration::from_secs(60);
                wait_recv(self, &mut batch, key, deadline)?;
                self.qps[&key]
                    .write_imm(local_ping, 8, lkey, remote_ping, end.rkey, PING_IMM)
                    .map_err(wrap_verbs)?;
                self.ensure_recv(key)?;
            }
        }
        Ok(rtts)
    }

    pub fn qp_count(&self) -> usize {
        self.qps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imm_encoding() {
        assert_eq!(imm(0, 3), 0x0003);
        assert_eq!(imm(0x1234, 0x5a), 0x12345a);
        assert_eq!(imm(0xffff, 0xff), 0xffffff);
    }

    #[test]
    fn expected_mask_shape() {
        // rank 0: succ 1, pred 3, forwarded src 2
        let expected = (1u32 << 1) | (1 << 3) | (1 << 2);
        assert_eq!(expected, 0b1110);
        // rank 2: succ 3, pred 1, forwarded src 0
        let expected = (1u32 << 3) | (1 << 1) | (1 << 0);
        assert_eq!(expected, 0b1011);
    }

    #[test]
    fn rank_ordered_sum_is_left_to_right() {
        // the property the gate checks: same inputs, same order -> same bits
        let a: Vec<f32> = vec![1.0, 1e20, -1e20, 1e-40];
        let b: Vec<f32> = vec![1e20, 1.0, 1e20, 1.0];
        let mut s1 = a[0] + a[1] + a[2] + a[3];
        let mut s2 = a[0] + a[2] + a[1] + a[3];
        assert_ne!(s1.to_bits(), s2.to_bits()); // order matters for f32
        s1 = 0.0;
        s2 = 0.0;
        for x in &a {
            s1 += x;
        }
        for x in &b {
            s2 += x;
        }
        assert_ne!(s1, s2);
    }
}
