//! Hand-written libibverbs FFI, transcribed field-for-field from the
//! rdma-core 50 headers on the Spark nodes
//! (`/usr/include/infiniband/verbs.h`, aarch64, glibc).
//!
//! Why not `rdma-sys`/bindgen: the verbs API surface this crate needs is
//! small (~20 functions, ~12 structs), the nodes have matching headers but
//! not necessarily a bindgen toolchain, and hand-transcribing from the real
//! header on the target keeps the build dependency-free and the layout
//! auditable. All struct definitions below are `#[repr(C)]` mirrors; enums
//! are `i32` constants (C enums are `int` on this ABI).
//!
//! Endianness: aarch64 is little-endian; `imm_data` is network byte order.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------- opaque

#[repr(C)]
pub struct ibv_device {
    _priv: [u8; 0],
}
#[repr(C)]
pub struct ibv_pd {
    _priv: [u8; 0],
}
/// `struct ibv_cq` — head field only (context at offset 0).
#[repr(C)]
pub struct ibv_cq {
    pub context: *mut ibv_context,
}
#[repr(C)]
pub struct ibv_comp_channel {
    _priv: [u8; 0],
}
#[repr(C)]
pub struct ibv_srq {
    _priv: [u8; 0],
}
#[repr(C)]
pub struct ibv_ah {
    _priv: [u8; 0],
}

/// The 32-entry function-pointer table at `ibv_context.ops` (rdma-core 50
/// order). Only the entries this crate calls through have real signatures;
/// the rest are slot placeholders to keep offsets exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvContextOps {
    pub _compat_query_device: *const c_void,
    pub _compat_query_port: *const c_void,
    pub _compat_alloc_pd: *const c_void,
    pub _compat_dealloc_pd: *const c_void,
    pub _compat_reg_mr: *const c_void,
    pub _compat_rereg_mr: *const c_void,
    pub _compat_dereg_mr: *const c_void,
    pub alloc_mw: *const c_void,
    pub bind_mw: *const c_void,
    pub dealloc_mw: *const c_void,
    pub _compat_create_cq: *const c_void,
    /// `ibv_poll_cq` is an inline wrapper over this slot in verbs.h
    pub poll_cq:
        Option<unsafe extern "C" fn(cq: *mut ibv_cq, num_entries: c_int, wc: *mut ibv_wc) -> c_int>,
    pub req_notify_cq: *const c_void,
    pub _compat_cq_event: *const c_void,
    pub _compat_resize_cq: *const c_void,
    pub _compat_destroy_cq: *const c_void,
    pub _compat_create_srq: *const c_void,
    pub _compat_modify_srq: *const c_void,
    pub _compat_query_srq: *const c_void,
    pub _compat_destroy_srq: *const c_void,
    pub post_srq_recv: *const c_void,
    pub _compat_create_qp: *const c_void,
    pub _compat_query_qp: *const c_void,
    pub _compat_modify_qp: *const c_void,
    pub _compat_destroy_qp: *const c_void,
    /// `ibv_post_send` is an inline wrapper over this slot
    pub post_send: Option<
        unsafe extern "C" fn(
            qp: *mut ibv_qp,
            wr: *mut ibv_send_wr,
            bad_wr: *mut *mut ibv_send_wr,
        ) -> c_int,
    >,
    /// `ibv_post_recv` is an inline wrapper over this slot
    pub post_recv: Option<
        unsafe extern "C" fn(
            qp: *mut ibv_qp,
            wr: *mut ibv_recv_wr,
            bad_wr: *mut *mut ibv_recv_wr,
        ) -> c_int,
    >,
    pub _compat_create_ah: *const c_void,
    pub _compat_destroy_ah: *const c_void,
    pub _compat_attach_mcast: *const c_void,
    pub _compat_detach_mcast: *const c_void,
    pub _compat_async_event: *const c_void,
}

/// `struct ibv_context` — head fields only (we read device and the ops
/// table; the tail — cmd_fd, mutex, abi_compat — is never touched).
#[repr(C)]
pub struct ibv_context {
    pub device: *mut ibv_device,
    pub ops: IbvContextOps,
}

// ------------------------------------------------------------- ibv_gid

#[repr(C)]
#[derive(Clone, Copy)]
pub union ibv_gid {
    pub raw: [u8; 16],
    pub global: IbvGidGlobal,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvGidGlobal {
    pub subnet_prefix: u64,
    pub interface_id: u64,
}

// ------------------------------------------------------------- attrs

/// `struct ibv_port_attr` (rdma-core 50).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ibv_port_attr {
    pub state: i32,
    pub max_mtu: i32,
    pub active_mtu: i32,
    pub gid_tbl_len: c_int,
    pub port_cap_flags: u32,
    pub max_msg_sz: u32,
    pub bad_pkey_cntr: u32,
    pub qkey_viol_cntr: u32,
    pub pkey_tbl_len: u16,
    pub lid: u16,
    pub sm_lid: u16,
    pub lmc: u8,
    pub max_vl_num: u8,
    pub sm_sl: u8,
    pub subnet_timeout: u8,
    pub init_type_reply: u8,
    pub active_width: u8,
    pub active_speed: u8,
    pub phys_state: u8,
    pub link_layer: u8,
    pub flags: u8,
    pub port_cap_flags2: u16,
    pub active_speed_ex: u32,
}

/// `struct ibv_global_route` (rdma-core 50).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_global_route {
    pub dgid: ibv_gid,
    pub flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
}

/// `struct ibv_ah_attr` — note rdma-core 50 puts `grh` FIRST.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_ah_attr {
    pub grh: ibv_global_route,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub static_rate: u8,
    pub is_global: u8,
    pub port_num: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ibv_qp_cap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

/// `struct ibv_qp_init_attr` (rdma-core 50).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_qp_init_attr {
    pub qp_context: *mut c_void,
    pub send_cq: *mut ibv_cq,
    pub recv_cq: *mut ibv_cq,
    pub srq: *mut ibv_srq,
    pub cap: ibv_qp_cap,
    pub qp_type: i32, // IBV_QPT_RC = 2, IBV_QPT_UC = 3, IBV_QPT_UD = 4
    pub sq_sig_all: c_int,
}

/// `struct ibv_qp_attr` (rdma-core 50) — transcribed exactly; note this
/// layout differs from older libibverbs (qp_access_flags sits before cap,
/// and grh is first inside ah_attr).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_qp_attr {
    pub qp_state: i32,
    pub cur_qp_state: i32,
    pub path_mtu: i32,
    pub path_mig_state: i32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: ibv_qp_cap,
    pub ah_attr: ibv_ah_attr,
    pub alt_ah_attr: ibv_ah_attr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    pub rate_limit: u32,
}

// ------------------------------------------------------------- work reqs

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ibv_sge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendRdma {
    pub remote_addr: u64,
    pub rkey: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendAtomic {
    pub remote_addr: u64,
    pub compare_add: u64,
    pub swap: u64,
    pub rkey: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendUd {
    pub ah: *mut ibv_ah,
    pub remote_qpn: u32,
    pub remote_qkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendWrBody {
    pub rdma: IbvSendRdma,
    pub atomic: IbvSendAtomic,
    pub ud: IbvSendUd,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendXrc {
    pub remote_srqn: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvMwBindInfo {
    pub mr: *mut ibv_mr,
    pub addr: u64,
    pub length: u64,
    pub access: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendBindMw {
    pub mw: *mut c_void,
    pub rkey: u32,
    pub bind_info: IbvMwBindInfo,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSendTso {
    pub hdr: *mut c_void,
    pub hdr_sz: u16,
    pub mss: u16,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendExt {
    pub bind_mw: IbvSendBindMw,
    pub tso: IbvSendTso,
}

/// `struct ibv_send_wr` (rdma-core 50), unions transcribed verbatim.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_send_wr {
    pub wr_id: u64,
    pub next: *mut ibv_send_wr,
    pub sg_list: *mut ibv_sge,
    pub num_sge: c_int,
    pub opcode: i32,
    pub send_flags: u32,
    pub imm_data: u32, // network byte order
    pub wr: IbvSendWrBody,
    pub qp_type: IbvSendXrc,
    pub ext: IbvSendExt,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_recv_wr {
    pub wr_id: u64,
    pub next: *mut ibv_recv_wr,
    pub sg_list: *mut ibv_sge,
    pub num_sge: c_int,
}

/// `struct ibv_wc` (rdma-core 50).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ibv_wc {
    pub wr_id: u64,
    pub status: i32,
    pub opcode: i32,
    pub vendor_err: u32,
    pub byte_len: u32,
    /// IBV_WC_WITH_IMM: immediate data, network byte order
    pub imm_data: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub sl: u8,
    pub dlid_path_bits: u8,
}

// ------------------------------------------------------------- resources

/// `struct ibv_mr` — only the head fields we read; rdma-core 50 has no tail
/// beyond rkey.
#[repr(C)]
pub struct ibv_mr {
    pub context: *mut ibv_context,
    pub pd: *mut ibv_pd,
    pub addr: *mut c_void,
    pub length: usize,
    pub handle: u32,
    pub lkey: u32,
    pub rkey: u32,
}

/// `struct ibv_qp` — head fields only (qp_num at offset 48).
#[repr(C)]
pub struct ibv_qp {
    pub context: *mut ibv_context,
    pub qp_context: *mut c_void,
    pub pd: *mut ibv_pd,
    pub send_cq: *mut ibv_cq,
    pub recv_cq: *mut ibv_cq,
    pub srq: *mut ibv_srq,
    pub handle: u32,
    pub qp_num: u32,
}

// ---------------------------------------------------------------- enums

// ibv_qp_type — IB order: SMI=0, GSI=1, RC=2 (verbs.h)
pub const IBV_QPT_RC: i32 = 2;
// ibv_qp_state
pub const IBV_QPS_INIT: i32 = 1;
pub const IBV_QPS_RTR: i32 = 2;
pub const IBV_QPS_RTS: i32 = 3;
pub const IBV_QPS_ERR: i32 = 6;
// ibv_mtu
pub const IBV_MTU_4096: i32 = 5;
// ibv_port_state
pub const IBV_PORT_ACTIVE: i32 = 4;
// ibv_link_layer
pub const IBV_LINK_LAYER_ETHERNET: u8 = 2;
// ibv_access_flags
pub const IBV_ACCESS_LOCAL_WRITE: u32 = 1;
pub const IBV_ACCESS_REMOTE_WRITE: u32 = 1 << 1;
// ibv_send_flags
pub const IBV_SEND_SIGNALED: u32 = 1 << 1;
// ibv_wr_opcode
pub const IBV_WR_RDMA_WRITE: i32 = 0;
pub const IBV_WR_RDMA_WRITE_WITH_IMM: i32 = 1;
// ibv_wc_opcode
pub const IBV_WC_SEND: i32 = 0;
pub const IBV_WC_RECV: i32 = 1 << 7;
pub const IBV_WC_RECV_RDMA_WITH_IMM: i32 = (1 << 7) + 1;
// ibv_wc_flags
pub const IBV_WC_WITH_IMM: u32 = 1 << 3;
// ibv_qp_attr_mask
pub const IBV_QP_STATE: i32 = 1 << 0;
pub const IBV_QP_ACCESS_FLAGS: i32 = 1 << 3;
pub const IBV_QP_PKEY_INDEX: i32 = 1 << 4;
pub const IBV_QP_PORT: i32 = 1 << 5;
pub const IBV_QP_AV: i32 = 1 << 7;
pub const IBV_QP_PATH_MTU: i32 = 1 << 8;
pub const IBV_QP_TIMEOUT: i32 = 1 << 9;
pub const IBV_QP_RETRY_CNT: i32 = 1 << 10;
pub const IBV_QP_RNR_RETRY: i32 = 1 << 11;
pub const IBV_QP_RQ_PSN: i32 = 1 << 12;
pub const IBV_QP_MAX_QP_RD_ATOMIC: i32 = 1 << 13;
pub const IBV_QP_MIN_RNR_TIMER: i32 = 1 << 15;
pub const IBV_QP_SQ_PSN: i32 = 1 << 16;
pub const IBV_QP_MAX_DEST_RD_ATOMIC: i32 = 1 << 17;
pub const IBV_QP_DEST_QPN: i32 = 1 << 20;

// ---------------------------------------------------------------- fns

#[link(name = "ibverbs")]
extern "C" {
    pub fn ibv_get_device_list(n: *mut c_int) -> *mut *mut ibv_device;
    pub fn ibv_free_device_list(list: *mut *mut ibv_device);
    pub fn ibv_get_device_name(device: *mut ibv_device) -> *const c_char;
    pub fn ibv_open_device(device: *mut ibv_device) -> *mut ibv_context;
    pub fn ibv_close_device(context: *mut ibv_context) -> c_int;
    pub fn ibv_query_port(
        context: *mut ibv_context,
        port_num: u8,
        attr: *mut ibv_port_attr,
    ) -> c_int;
    pub fn ibv_query_gid(
        context: *mut ibv_context,
        port_num: u8,
        index: c_int,
        gid: *mut ibv_gid,
    ) -> c_int;
    pub fn ibv_alloc_pd(context: *mut ibv_context) -> *mut ibv_pd;
    pub fn ibv_dealloc_pd(pd: *mut ibv_pd) -> c_int;
    pub fn ibv_reg_mr(
        pd: *mut ibv_pd,
        addr: *mut c_void,
        length: usize,
        access: c_int,
    ) -> *mut ibv_mr;
    pub fn ibv_dereg_mr(mr: *mut ibv_mr) -> c_int;
    pub fn ibv_create_cq(
        context: *mut ibv_context,
        cqe: c_int,
        cq_context: *mut c_void,
        channel: *mut ibv_comp_channel,
        comp_vector: c_int,
    ) -> *mut ibv_cq;
    pub fn ibv_destroy_cq(cq: *mut ibv_cq) -> c_int;
    pub fn ibv_create_qp(pd: *mut ibv_pd, qp_init_attr: *mut ibv_qp_init_attr) -> *mut ibv_qp;
    pub fn ibv_modify_qp(qp: *mut ibv_qp, attr: *mut ibv_qp_attr, attr_mask: c_int) -> c_int;
    pub fn ibv_destroy_qp(qp: *mut ibv_qp) -> c_int;
}

pub fn wc_status_name(status: i32) -> &'static str {
    match status {
        0 => "SUCCESS",
        1 => "LOC_LEN_ERR",
        2 => "LOC_QP_OP_ERR",
        3 => "LOC_EEC_OP_ERR",
        4 => "LOC_PROT_ERR",
        5 => "WR_FLUSH_ERR",
        6 => "MW_BIND_ERR",
        7 => "BAD_RESP_ERR",
        8 => "LOC_ACCESS_ERR",
        9 => "REM_INV_REQ_ERR",
        10 => "REM_ACCESS_ERR",
        11 => "REM_OP_ERR",
        12 => "RETRY_EXC_ERR",
        13 => "RNR_RETRY_EXC_ERR",
        14 => "LOC_RDD_VIOL_ERR",
        15 => "REM_INV_RD_REQ_ERR",
        16 => "REM_ABORT_ERR",
        17 => "INV_EECN_ERR",
        18 => "INV_EEC_STATE_ERR",
        19 => "FATAL_ERR",
        20 => "RESP_TIMEOUT_ERR",
        21 => "GENERAL_ERR",
        _ => "UNKNOWN",
    }
}
