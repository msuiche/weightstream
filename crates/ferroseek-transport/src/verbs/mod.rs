//! Thin safe wrapper over libibverbs for the ring transport.
//!
//! Scope: RC QPs over RoCE v2, one QP per (neighbor, rail), a single
//! registered host buffer per rank, one CQ per HCA. Discovery maps each
//! RoCE interface (by its IPv4) to the (device, port, gid_index) triple
//! that serves it, so the node map's interface names never need to appear
//! in ibverbs' device naming.

mod ffi;

pub use ffi::{
    IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_WRITE, IBV_LINK_LAYER_ETHERNET, IBV_MTU_4096,
    IBV_PORT_ACTIVE, IBV_QP_ACCESS_FLAGS, IBV_QP_AV, IBV_QP_DEST_QPN, IBV_QP_MAX_DEST_RD_ATOMIC,
    IBV_QP_MAX_QP_RD_ATOMIC, IBV_QP_MIN_RNR_TIMER, IBV_QP_PATH_MTU, IBV_QP_PKEY_INDEX, IBV_QP_PORT,
    IBV_QP_RETRY_CNT, IBV_QP_RNR_RETRY, IBV_QP_RQ_PSN, IBV_QP_SQ_PSN, IBV_QP_STATE, IBV_QP_TIMEOUT,
    IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS, IBV_QPT_RC, IBV_SEND_SIGNALED, IBV_WC_RECV,
    IBV_WC_RECV_RDMA_WITH_IMM, IBV_WC_SEND, IBV_WC_WITH_IMM, IBV_WR_RDMA_WRITE_WITH_IMM,
    ibv_ah_attr, ibv_gid, ibv_global_route, ibv_qp_attr, ibv_qp_init_attr, ibv_send_wr, ibv_sge,
};

use std::collections::BTreeMap;
use std::ffi::{CStr, c_int};
use std::os::raw::c_void;

#[derive(Debug)]
pub struct VerbsError(pub String);
impl std::fmt::Display for VerbsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "verbs: {}", self.0)
    }
}
impl std::error::Error for VerbsError {}
pub type VerbsResult<T> = Result<T, VerbsError>;

fn err<T>(ctx: &str) -> VerbsResult<T> {
    Err(VerbsError(format!(
        "{ctx}: {}",
        std::io::Error::last_os_error()
    )))
}

/// IPv4 embedded in a RoCE v2 GID (::ffff:a.b.c.d form), if any.
fn gid_roce_v2_ip(gid: &ibv_gid) -> Option<std::net::Ipv4Addr> {
    unsafe {
        let r = &gid.raw;
        (r[10] == 0xff && r[11] == 0xff && r[0..10].iter().all(|&b| b == 0))
            .then(|| std::net::Ipv4Addr::new(r[12], r[13], r[14], r[15]))
    }
}

/// One opened HCA: context, protection domain, completion queue.
pub struct DeviceCtx {
    /// ibverbs device name, e.g. "mlx5_0" (sysfs hw_counters live under this)
    pub name: String,
    ctx: *mut ffi::ibv_context,
    pd: *mut ffi::ibv_pd,
    cq: *mut ffi::ibv_cq,
}

impl DeviceCtx {
    pub fn cq_ptr(&self) -> *mut ffi::ibv_cq {
        self.cq
    }
}

/// All HCAs on this node, opened once.
pub struct Verbs {
    pub devices: Vec<DeviceCtx>,
}

/// (device index, port, gid index) serving one rail: the port whose RoCE v2
/// GID carries an IPv4 inside `subnet`. `ip` is that address; `active_mtu`
/// drives the QP path MTU (IB enum units).
#[derive(Clone, Debug)]
pub struct PortRef {
    pub dev: usize,
    pub port: u8,
    pub gid_index: u8,
    pub gid: [u8; 16],
    pub ip: String,
    pub active_mtu: i32,
}

impl Verbs {
    pub fn open() -> VerbsResult<Verbs> {
        unsafe {
            let mut n: i32 = 0;
            let list = ffi::ibv_get_device_list(&mut n);
            if list.is_null() || n == 0 {
                if !list.is_null() {
                    ffi::ibv_free_device_list(list);
                }
                return Err(VerbsError("no RDMA devices found".into()));
            }
            let mut devices = Vec::new();
            for i in 0..n as usize {
                let dev = *list.add(i);
                let name = CStr::from_ptr(ffi::ibv_get_device_name(dev))
                    .to_string_lossy()
                    .into_owned();
                let ctx = ffi::ibv_open_device(dev);
                if ctx.is_null() {
                    continue;
                }
                let pd = ffi::ibv_alloc_pd(ctx);
                let cq =
                    ffi::ibv_create_cq(ctx, 4096, std::ptr::null_mut(), std::ptr::null_mut(), 0);
                if pd.is_null() || cq.is_null() {
                    if !cq.is_null() {
                        ffi::ibv_destroy_cq(cq);
                    }
                    if !pd.is_null() {
                        ffi::ibv_dealloc_pd(pd);
                    }
                    ffi::ibv_close_device(ctx);
                    continue;
                }
                devices.push(DeviceCtx { name, ctx, pd, cq });
            }
            ffi::ibv_free_device_list(list);
            if devices.is_empty() {
                return Err(VerbsError("no RDMA device could be opened".into()));
            }
            Ok(Verbs { devices })
        }
    }

    /// Find the port whose RoCE v2 GID carries an IPv4 inside `subnet`,
    /// on the ibverbs device named `ibdev` (the node map's interface
    /// entries are ibverbs device names — the netdev names can differ).
    /// `prefer_gid` is tried first (the map's roce_gid_index), then the
    /// whole GID table is scanned.
    pub fn find_port_in_subnet(
        &self,
        ibdev: &str,
        subnet: &str,
        prefer_gid: u8,
    ) -> VerbsResult<PortRef> {
        unsafe {
            let mut found = None;
            for (di, d) in self.devices.iter().enumerate() {
                if d.name != ibdev {
                    continue;
                }
                for port in 1..=2u8 {
                    let mut pa = ffi::ibv_port_attr::default();
                    if ffi::ibv_query_port(d.ctx, port, &mut pa) != 0 {
                        continue;
                    }
                    if pa.state != IBV_PORT_ACTIVE || pa.link_layer != IBV_LINK_LAYER_ETHERNET {
                        continue;
                    }
                    let mut first = None;
                    for gi in 0..pa.gid_tbl_len {
                        let mut gid = ffi::ibv_gid { raw: [0u8; 16] };
                        if ffi::ibv_query_gid(d.ctx, port, gi, &mut gid) != 0 {
                            continue;
                        }
                        let Some(ip) = gid_roce_v2_ip(&gid) else {
                            continue;
                        };
                        if !crate::nodes::in_subnet(&ip.to_string(), subnet) {
                            continue;
                        }
                        let pr = PortRef {
                            dev: di,
                            port,
                            gid_index: gi as u8,
                            gid: gid.raw,
                            ip: ip.to_string(),
                            active_mtu: pa.active_mtu,
                        };
                        if gi as u8 == prefer_gid {
                            return Ok(pr);
                        }
                        if first.is_none() {
                            first = Some(pr);
                        }
                    }
                    if found.is_none() {
                        found = first;
                    }
                }
            }
            found.ok_or_else(|| {
                VerbsError(format!(
                    "no RoCE v2 GID on device {ibdev} carries an address in {subnet}"
                ))
            })
        }
    }

    /// First active Ethernet (RoCE) port on device `d` — for the
    /// single-node QP loopback smoke test.
    pub fn probe_first_port(&self, d: usize) -> VerbsResult<PortRef> {
        unsafe {
            let dev = &self.devices[d];
            let mut fallback = None;
            for port in 1..=2u8 {
                let mut pa = ffi::ibv_port_attr::default();
                if ffi::ibv_query_port(dev.ctx, port, &mut pa) != 0 {
                    continue;
                }
                if pa.state != IBV_PORT_ACTIVE || pa.link_layer != IBV_LINK_LAYER_ETHERNET {
                    continue;
                }
                for gi in 0..pa.gid_tbl_len {
                    let mut gid = ffi::ibv_gid { raw: [0u8; 16] };
                    if ffi::ibv_query_gid(dev.ctx, port, gi, &mut gid) != 0 {
                        continue;
                    }
                    let r = gid.raw;
                    // prefer an IPv4-mapped RoCE v2 GID; fall back to any
                    if r[10] == 0xff && r[11] == 0xff {
                        return Ok(PortRef {
                            dev: d,
                            port,
                            gid_index: gi as u8,
                            gid: r,
                            ip: format!("{}.{}.{}.{}", r[12], r[13], r[14], r[15]),
                            active_mtu: pa.active_mtu,
                        });
                    }
                    if fallback.is_none() && r.iter().any(|&b| b != 0) {
                        fallback = Some(PortRef {
                            dev: d,
                            port,
                            gid_index: gi as u8,
                            gid: r,
                            ip: String::new(),
                            active_mtu: pa.active_mtu,
                        });
                    }
                }
            }
            fallback.ok_or_else(|| VerbsError("no usable gid on device".into()))
        }
    }

    /// Register host memory on one device. One buffer registered per
    /// distinct HCA the rank's links use — the rkey a peer needs depends on
    /// which of my ports the write lands on.
    pub fn register(&self, dev: usize, buf: &mut [u8]) -> VerbsResult<MrInfo> {
        unsafe {
            let d = &self.devices[dev];
            let mr = ffi::ibv_reg_mr(
                d.pd,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) as i32,
            );
            if mr.is_null() {
                return err("ibv_reg_mr");
            }
            Ok(MrInfo {
                dev,
                lkey: (*mr).lkey,
                rkey: (*mr).rkey,
                _mr: mr,
            })
        }
    }

    /// Create an RC QP in INIT state on (dev, port).
    pub fn create_qp(&self, p: &PortRef) -> VerbsResult<Qp> {
        unsafe {
            let d = &self.devices[p.dev];
            let mut init = ibv_qp_init_attr {
                qp_context: std::ptr::null_mut(),
                send_cq: d.cq,
                recv_cq: d.cq,
                srq: std::ptr::null_mut(),
                cap: ffi::ibv_qp_cap {
                    max_send_wr: 256,
                    max_recv_wr: 512,
                    max_send_sge: 4,
                    max_recv_sge: 4,
                    max_inline_data: 0,
                },
                qp_type: IBV_QPT_RC,
                sq_sig_all: 1,
            };
            let qp = ffi::ibv_create_qp(d.pd, &mut init);
            if qp.is_null() {
                return err("ibv_create_qp");
            }
            // INIT (this mlx5/kernel requires PKEY_INDEX in the mask)
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = IBV_QPS_INIT;
            attr.port_num = p.port;
            attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            let mask = IBV_QP_STATE | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS | IBV_QP_PKEY_INDEX;
            if ffi::ibv_modify_qp(qp, &mut attr, mask) != 0 {
                ffi::ibv_destroy_qp(qp);
                return err("ibv_modify_qp INIT");
            }
            Ok(Qp {
                dev: p.dev,
                qpn: (*qp).qp_num,
                local_port: p.port,
                local_gid_index: p.gid_index,
                path_mtu: if p.active_mtu >= 1 {
                    p.active_mtu
                } else {
                    IBV_MTU_4096
                },
                qp,
            })
        }
    }
}

/// Registered memory region metadata (one per HCA the buffer is exposed on).
pub struct MrInfo {
    pub dev: usize,
    pub lkey: u32,
    pub rkey: u32,
    _mr: *mut ffi::ibv_mr,
}

/// An RC QP in INIT state, bound to a device+port+gid.
pub struct Qp {
    pub dev: usize,
    pub qpn: u32,
    pub local_port: u8,
    pub local_gid_index: u8,
    path_mtu: i32,
    qp: *mut ffi::ibv_qp,
}

impl Qp {
    /// Transition to RTR against a peer endpoint.
    pub fn to_rtr(&self, peer_qpn: u32, peer_gid: [u8; 16]) -> VerbsResult<()> {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = IBV_QPS_RTR;
            attr.path_mtu = self.path_mtu;
            attr.dest_qp_num = peer_qpn;
            attr.rq_psn = 0;
            attr.min_rnr_timer = 12;
            attr.max_dest_rd_atomic = 1;
            attr.ah_attr = ibv_ah_attr {
                grh: ibv_global_route {
                    dgid: ffi::ibv_gid { raw: peer_gid },
                    flow_label: 0,
                    sgid_index: self.local_gid_index,
                    hop_limit: 1, // direct cable, one /24, no routing
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 1, // RoCE always uses the GRH
                port_num: self.local_port,
            };
            let mask = IBV_QP_STATE
                | IBV_QP_AV
                | IBV_QP_PATH_MTU
                | IBV_QP_DEST_QPN
                | IBV_QP_RQ_PSN
                | IBV_QP_MIN_RNR_TIMER
                | IBV_QP_MAX_DEST_RD_ATOMIC;
            if ffi::ibv_modify_qp(self.qp, &mut attr, mask) != 0 {
                return err("ibv_modify_qp RTR");
            }
            Ok(())
        }
    }

    /// Transition to RTS.
    pub fn to_rts(&self) -> VerbsResult<()> {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 3;
            attr.sq_psn = 0;
            attr.max_rd_atomic = 1;
            let mask = IBV_QP_STATE
                | IBV_QP_TIMEOUT
                | IBV_QP_RETRY_CNT
                | IBV_QP_RNR_RETRY
                | IBV_QP_SQ_PSN
                | IBV_QP_MAX_QP_RD_ATOMIC;
            if ffi::ibv_modify_qp(self.qp, &mut attr, mask) != 0 {
                return err("ibv_modify_qp RTS");
            }
            Ok(())
        }
    }

    /// RDMA write-with-imm of `len` bytes from `src` to peer's `remote_addr`
    /// with `rkey`, immediate `imm` (host order). Returns after posting.
    pub fn write_imm(
        &self,
        src: u64,
        len: u32,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        imm: u32,
    ) -> VerbsResult<()> {
        unsafe {
            let mut sge = ibv_sge {
                addr: src,
                length: len,
                lkey,
            };
            let mut wr = std::mem::zeroed::<ibv_send_wr>();
            wr.wr_id = 0;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.imm_data = imm.to_be();
            wr.wr.rdma = ffi::IbvSendRdma { remote_addr, rkey };
            let mut bad: *mut ibv_send_wr = std::ptr::null_mut();
            // rdma-core 50: ibv_post_send is an inline wrapper over the
            // context ops table
            let post_send = (*(*self.qp).context)
                .ops
                .post_send
                .ok_or_else(|| VerbsError("provider has no post_send op".into()))?;
            if post_send(self.qp, &mut wr, &mut bad) != 0 {
                return err("ibv_post_send RDMA_WRITE_WITH_IMM");
            }
            Ok(())
        }
    }

    /// Post `n` receive work requests (empty sg_list: RDMA write-with-imm
    /// places data at the sender's remote_addr; the recv WQE only carries
    /// the immediate).
    pub fn post_recv(&self, n: usize) -> VerbsResult<()> {
        unsafe {
            let mut wr = std::mem::zeroed::<ffi::ibv_recv_wr>();
            wr.sg_list = std::ptr::null_mut();
            wr.num_sge = 0;
            let mut bad: *mut ffi::ibv_recv_wr = std::ptr::null_mut();
            let post_recv = (*(*self.qp).context)
                .ops
                .post_recv
                .ok_or_else(|| VerbsError("provider has no post_recv op".into()))?;
            for _ in 0..n {
                if post_recv(self.qp, &mut wr, &mut bad) != 0 {
                    return err("ibv_post_recv");
                }
            }
            Ok(())
        }
    }

    pub fn to_err(&self) {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ffi::IBV_QPS_ERR;
            let _ = ffi::ibv_modify_qp(self.qp, &mut attr, IBV_QP_STATE);
        }
    }
}

impl Drop for Qp {
    fn drop(&mut self) {
        unsafe { ffi::ibv_destroy_qp(self.qp) };
    }
}
impl Drop for MrInfo {
    fn drop(&mut self) {
        unsafe { ffi::ibv_dereg_mr(self._mr) };
    }
}
impl Drop for DeviceCtx {
    fn drop(&mut self) {
        unsafe {
            ffi::ibv_destroy_cq(self.cq);
            ffi::ibv_dealloc_pd(self.pd);
            ffi::ibv_close_device(self.ctx);
        }
    }
}

/// One CQ poll batch, including inline completion storage.
pub struct PollBatch {
    wcs: [ffi::ibv_wc; 16],
    events: [PollEvent; 16],
}

impl PollBatch {
    pub fn poll(&mut self, cq: *mut ffi::ibv_cq) -> VerbsResult<usize> {
        unsafe {
            // rdma-core 50: ibv_poll_cq is an inline wrapper over the
            // context ops table
            let poll_cq = (*(*cq).context)
                .ops
                .poll_cq
                .ok_or_else(|| VerbsError("provider has no poll_cq op".into()))?;
            let n = poll_cq(cq, self.wcs.len() as c_int, self.wcs.as_mut_ptr());
            if n < 0 {
                return err("ibv_poll_cq");
            }
            for (event, wc) in self.events.iter_mut().zip(&self.wcs[..n as usize]) {
                if wc.status != 0 {
                    return Err(VerbsError(format!(
                        "CQE error: {} (vendor {})",
                        ffi::wc_status_name(wc.status),
                        wc.vendor_err
                    )));
                }
                event.kind = if wc.opcode & IBV_WC_RECV != 0 {
                    PollKind::Recv {
                        imm: u32::from_be(wc.imm_data),
                    }
                } else {
                    PollKind::Send
                };
                event.qp_num = wc.qp_num;
            }
            Ok(n as usize)
        }
    }

    pub fn event(&self, index: usize) -> PollEvent {
        self.events[index]
    }
}

impl Default for PollBatch {
    fn default() -> Self {
        PollBatch {
            wcs: unsafe { [std::mem::zeroed::<ffi::ibv_wc>(); 16] },
            events: [PollEvent {
                kind: PollKind::Send,
                qp_num: 0,
            }; 16],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PollKind {
    /// receive completion; imm already converted to host order
    Recv {
        imm: u32,
    },
    Send,
}

#[derive(Debug, Clone, Copy)]
pub struct PollEvent {
    pub kind: PollKind,
    pub qp_num: u32,
}

/// Map qpn -> (peer, rail), built by the ring so poll events can be
/// attributed without touching the QP handle.
pub type QpnMap = BTreeMap<u32, (u8, usize)>;
