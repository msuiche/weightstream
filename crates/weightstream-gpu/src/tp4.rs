//! TP4 wiring (docs/transport.md): the one-shot RoCE ring as the engine's
//! [`Collectives`], plus the 4-rank rendezvous the CLI drives.
//!
//! Flow per rank process: parse the node map, build the ring QPs
//! (`Ring::new`), rendezvous with the head over the management TCP
//! (`Linkup` → `Rtr` → `Rts`), then return — the engine loads its shard,
//! every rank reports `loaded`, the head releases the decode, and the
//! ring's collectives carry the three per-layer all-reduces and the logits
//! all-gather.
//!
//! The fail-stop deadline is generous by engine standards (default 60 s):
//! between collectives a rank may spend seconds staging its 96-expert
//! slice from cold disk, so the bench's 100 ms budget would abort a
//! healthy step. The abort still fires on a dead peer.

use std::time::Duration;

use weightstream_transport::ctrl::{Head, Phase, Worker};
use weightstream_transport::nodes::NodesFile;
use weightstream_transport::ring::Ring;
use weightstream_transport::verbs::Verbs;

use crate::engine::Collectives;
use std::io;

pub struct RingComm {
    ring: Ring,
}

impl Collectives for RingComm {
    fn allreduce(&mut self, input: &[f32]) -> io::Result<Vec<f32>> {
        let out = self
            .ring
            .allreduce(input)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("ring allreduce: {e}")))?;
        Ok(out.to_vec())
    }

    fn allgather(&mut self, contribution: &[f32]) -> io::Result<Vec<f32>> {
        let out = self
            .ring
            .allgather(contribution)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("ring allgather: {e}")))?;
        Ok(out.to_vec())
    }

    fn stats(&self) -> String {
        format!(
            "ring: {} collectives, {:.1} ms total, {:.0} us avg",
            self.ring.coll_count,
            self.ring.coll_us / 1000.0,
            if self.ring.coll_count > 0 {
                self.ring.coll_us / self.ring.coll_count as f64
            } else {
                0.0
            }
        )
    }
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    unsafe {
        if libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) != 0 {
            return String::new();
        }
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

enum Ctrl {
    Head(Head),
    Worker(Worker),
}

/// One rank's TP4 session: the ring (as the engine's [`Collectives`]) and
/// the live management connection for the post-load start barrier. The
/// engine builder takes the comm out; the session keeps the control
/// channel alive until the decode finishes.
pub struct Session {
    comm: Option<Box<dyn Collectives>>,
    pub rank: u8,
    ctrl: Ctrl,
    elems: usize,
}

impl Session {
    /// Hand the ring to the engine (once).
    pub fn take_comm(&mut self) -> Box<dyn Collectives> {
        self.comm
            .take()
            .expect("tp4 comm handed to the engine once")
    }
}

impl Session {
    /// Every rank calls this once its model shard is loaded; returns when
    /// all four are ready. The head collects the workers' Ready messages
    /// and releases everyone with a `Start` go.
    pub fn start_barrier(&mut self) -> io::Result<()> {
        match &mut self.ctrl {
            Ctrl::Head(h) => {
                h.collect_ready(Duration::from_secs(600))
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                h.send_go(Phase::Start, serde_json::json!({}))
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            }
            Ctrl::Worker(w) => {
                w.send_ready(serde_json::json!({ "rank": self.rank, "loaded": true }))
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                match w
                    .wait_go()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
                {
                    Some((Phase::Start, _)) => {}
                    Some((p, _)) => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("expected Start, got {p:?}"),
                        ));
                    }
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "head closed the control channel before Start",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Serving barrier: the head broadcasts the next request (prompt text
    /// and token budget); workers block for it. `Ok(None)` on the worker
    /// side means the head closed (Bye) — the caller exits.
    pub fn next_request(
        &mut self,
        request: Option<serde_json::Value>,
    ) -> io::Result<Option<serde_json::Value>> {
        match &mut self.ctrl {
            Ctrl::Head(h) => match request {
                Some(req) => {
                    h.send_go(Phase::Next, req.clone())
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    h.collect_ready(Duration::from_secs(600))
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    Ok(Some(req))
                }
                None => {
                    h.send_bye();
                    Ok(None)
                }
            },
            Ctrl::Worker(w) => match w
                .wait_go()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            {
                Some((Phase::Next, cmd)) => {
                    w.send_ready(serde_json::json!({ "rank": self.rank }))
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    Ok(Some(cmd))
                }
                Some((p, _)) => Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("expected Next, got {p:?}"),
                )),
                None => Ok(None), // head said Bye: done
            },
        }
    }
    /// Clean shutdown: the head says bye; workers are done when the
    /// process exits (decode has finished by then).
    pub fn finish(&mut self) {
        if let Ctrl::Head(h) = &mut self.ctrl {
            h.send_bye();
        }
    }

    pub fn elems(&self) -> usize {
        self.elems
    }
}

/// Join the 4-rank ring: QP bring-up and rendezvous over the management
/// network (the transport bench's pattern, without the probe/gate/soak
/// phases — the engine itself is the gate). `elems` is the largest
/// collective payload (fp32 element count) the engine will issue; smaller
/// collectives shrink the wire payload, the slot stride stays fixed.
pub fn join(
    ring_path: &str,
    rank_override: Option<u8>,
    port: u16,
    elems: usize,
    wait_ms: u64,
) -> io::Result<Session> {
    let json = std::fs::read_to_string(ring_path)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("{ring_path}: {e}")))?;
    let map = NodesFile::parse(&json)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let me = map
        .my_rank(&hostname(), rank_override)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let my_node = map
        .node_of(me)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
        .clone();
    let cfg = map
        .ring_cfg(me)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    eprintln!(
        "tp4: rank {me} ({}) elems {elems} wait {wait_ms} ms",
        my_node.name
    );

    let verbs = Verbs::open().map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let (mut ring, my_ep) = Ring::new(verbs, &cfg, elems, Duration::from_millis(wait_ms))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let ctrl = if my_node.head {
        let mut head = Head::bind(&my_node.management_ip, port, me)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        head.wait_workers(4, Duration::from_secs(300))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        eprintln!("tp4 rank {me}: all 4 ranks rendezvoused");

        // Linkup: exchange QP endpoints
        head.send_go(Phase::Linkup, serde_json::json!({}))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut all = vec![(me, my_ep)];
        for data in head
            .collect_ready(Duration::from_secs(60))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
        {
            let r = data["rank"].as_u64().unwrap_or(255) as u8;
            all.push((r, data));
        }
        all.sort_by_key(|(r, _)| *r);
        if all.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("only {} ranks in the endpoint exchange", all.len()),
            ));
        }
        let ring_map: Vec<serde_json::Value> = all.into_iter().map(|(_, e)| e).collect();

        // RTR, then RTS with a barrier between
        head.send_go(Phase::Rtr, serde_json::json!({ "ring": ring_map.clone() }))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        ring.to_rtr(&serde_json::Value::Array(ring_map))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        head.collect_ready(Duration::from_secs(60))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        head.send_go(Phase::Rts, serde_json::json!({}))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        ring.to_rts()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        head.collect_ready(Duration::from_secs(60))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        eprintln!("tp4 rank {me}: QPs live (4 links x 2 rails)");
        Ctrl::Head(head)
    } else {
        let mut w = Worker::connect(&cfg.head_ip, port, me, Duration::from_secs(300))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        loop {
            let Some((phase, cmd)) = w
                .wait_go()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "head closed the control channel during rendezvous",
                ));
            };
            match phase {
                Phase::Linkup => {
                    w.send_ready(my_ep.clone())
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                }
                Phase::Rtr => {
                    ring.to_rtr(&cmd["ring"])
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                    w.send_ready(serde_json::json!({ "rank": me }))
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                }
                Phase::Rts => {
                    ring.to_rts()
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                    w.send_ready(serde_json::json!({ "rank": me }))
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    break;
                }
                p => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("unexpected phase {p:?} during rendezvous"),
                    ));
                }
            }
        }
        eprintln!("tp4 rank {me}: QPs live");
        Ctrl::Worker(w)
    };

    Ok(Session {
        comm: Some(Box::new(RingComm { ring })),
        rank: me,
        ctrl,
        elems,
    })
}
