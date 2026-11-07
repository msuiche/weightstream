//! Thin launch wrappers over the ferroseek-cuda kernels. One function per
//! kernel, hiding the `launch!` macro and grid arithmetic; every wrapper
//! enqueues on the global stream, so ordering follows issue order.

use cust::memory::{AsyncCopyDestination, DeviceBuffer, DeviceSlice};
use cust::prelude::*;
use ferroseek_cuda::gpu::Cuda;

pub type F32 = DevicePointer<f32>;
pub type U8 = DevicePointer<u8>;
pub type U16 = DevicePointer<u16>;
pub type I32 = DevicePointer<i32>;
pub type U32 = DevicePointer<u32>;
pub type I64 = DevicePointer<i64>;

/// Pointer to `buf[a..a+n]`.
pub fn row(buf: &DeviceBuffer<f32>, a: usize, n: usize) -> F32 {
    assert!(a + n <= buf.len());
    buf.index(a..a + n).as_device_ptr()
}

pub fn row_i32(buf: &DeviceBuffer<i32>, a: usize, n: usize) -> I32 {
    assert!(a + n <= buf.len());
    buf.index(a..a + n).as_device_ptr()
}

pub fn row_u8(buf: &DeviceBuffer<u8>, a: usize, n: usize) -> U8 {
    assert!(a + n <= buf.len());
    buf.index(a..a + n).as_device_ptr()
}

pub fn row_u16(buf: &DeviceBuffer<u16>, a: usize, n: usize) -> U16 {
    assert!(a + n <= buf.len());
    buf.index(a..a + n).as_device_ptr()
}

/// Base pointer of any device buffer.
pub fn base<T: cust::memory::DeviceCopy>(buf: &DeviceBuffer<T>) -> DevicePointer<T> {
    buf.as_device_ptr()
}

/// Pointer to `buf[a..a+n]` of a u16 buffer.
pub fn row16(buf: &DeviceBuffer<u16>, a: usize, n: usize) -> DevicePointer<u16> {
    assert!(a + n <= buf.len());
    buf.index(a..a + n).as_device_ptr()
}

pub fn base_u8(buf: &DeviceBuffer<u8>) -> U8 {
    buf.as_device_ptr()
}

pub fn base_u16(buf: &DeviceBuffer<u16>) -> U16 {
    buf.as_device_ptr()
}

pub fn base_i32(buf: &DeviceBuffer<i32>) -> I32 {
    buf.as_device_ptr()
}

pub fn base_i64(buf: &DeviceBuffer<i64>) -> I64 {
    buf.as_device_ptr()
}

fn gx(n: usize, threads: u32) -> u32 {
    n.div_ceil(threads as usize) as u32
}

/// `moe_local_slots` (moe_grouped.cu): device-side local slot selection
/// from the gate output — the host never reads the routing.
pub fn moe_local_slots(
    c: &Cuda,
    gate_idx: I32,
    local_slots: I32,
    local_experts: I32,
    local_count: I32,
    top_k: usize,
    e_start: usize,
    e_len: usize,
) {
    let f = c.moe_grouped.get_function("moe_local_slots").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<1, 1, 0, stream>>>(
                gate_idx,
                local_slots,
                local_experts,
                local_count,
                top_k as i32,
                e_start as i32,
                e_len as i32
            )
        )
        .unwrap()
    }
}

/// `expert_glu` (moe_grouped.cu): fused w1·x / w3·x dots + swiglu × route
/// weight for every local slot, one block per (slot, n).
#[allow(clippy::too_many_arguments)]
pub fn expert_glu(
    c: &Cuda,
    act_x: U8,
    act_x_s: U8,
    w1: U8,
    w1_s: U8,
    w3: U8,
    w3_s: U8,
    route_w: F32,
    local_slots: I32,
    local_experts: I32,
    local_count: I32,
    h_slots: F32,
    n: usize,
    k: usize,
    top_k: usize,
) {
    let f = c.moe_grouped.get_function("expert_glu").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(n as u32, top_k as u32), 256, 0, stream>>>(
                act_x, act_x_s, w1, w1_s, w3, w3_s, route_w, local_slots,
                local_experts, local_count, h_slots, n as i32, k as i32,
                top_k as i32
            )
        )
        .unwrap()
    }
}

/// `expert_quant` (moe_grouped.cu): act-quant emit per slot.
pub fn expert_quant(
    c: &Cuda,
    h_slots: F32,
    local_slots: I32,
    local_count: I32,
    q_slots: U8,
    s_slots: U8,
    n: usize,
    top_k: usize,
) {
    let f = c.moe_grouped.get_function("expert_quant").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<((n / 32) as u32, top_k as u32), 32, 0, stream>>>(
                h_slots, local_slots, local_count, q_slots, s_slots,
                n as i32, top_k as i32
            )
        )
        .unwrap()
    }
}

/// `expert_reduce` (moe_grouped.cu): y = Σ_slots-ascending w2·act_h —
/// same order and per-expert reduction shape as the host loop.
#[allow(clippy::too_many_arguments)]
pub fn expert_reduce(
    c: &Cuda,
    act_slots: U8,
    act_s_slots: U8,
    w2: U8,
    w2_s: U8,
    gate_idx: I32,
    local_slots: I32,
    local_count: I32,
    y: F32,
    dim: usize,
    inter: usize,
    top_k: usize,
    e_start: usize,
) {
    let f = c.moe_grouped.get_function("expert_reduce").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<dim as u32, 256, 0, stream>>>(
                act_slots, act_s_slots, w2, w2_s, gate_idx, local_slots,
                local_count, y, dim as i32, inter as i32, top_k as i32,
                e_start as i32
            )
        )
        .unwrap()
    }
}
/// `rmsnorm` (norm.cu): one block per row. Safe to call with `x == out`
/// (the reduction finishes before any element is written).
pub fn rmsnorm(c: &Cuda, x: F32, w: F32, out: F32, dim: usize, eps: f32) {
    let f = c.norm.get_function("rmsnorm").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<1u32, 256, 0, stream>>>(x, w, out, dim as i32, eps)).unwrap() }
}

/// `rmsnorm`, M rows: the kernel already indexes rows by blockIdx.x.
pub fn rmsnorm_m(c: &Cuda, x: F32, w: F32, out: F32, n_rows: usize, dim: usize, eps: f32) {
    let f = c.norm.get_function("rmsnorm").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<n_rows as u32, 256, 0, stream>>>(x, w, out, dim as i32, eps)).unwrap() }
}

/// `rope` (rope.cu): rows are positions (`pos_offset + row`).
pub fn rope(
    c: &Cuda,
    x: F32,
    cos: F32,
    sin: F32,
    n_rows: usize,
    dim: usize,
    rope_dim: usize,
    pos_offset: u64,
    conj: bool,
) {
    let f = c.rope.get_function("rope").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(gx(rope_dim / 2, 32), n_rows as u32), 32, 0, stream>>>(
                x, cos, sin, n_rows as i32, dim as i32, rope_dim as i32, pos_offset as i32, conj as i32
            )
        )
        .unwrap()
    }
}

/// `rope_uniform_pos` (rope.cu): every row at one position — the decode
/// q/o layouts are `[n_heads, head_dim]` for a single token.
pub fn rope_uniform_pos(
    c: &Cuda,
    x: F32,
    cos: F32,
    sin: F32,
    n_rows: usize,
    dim: usize,
    rope_dim: usize,
    pos: u64,
    conj: bool,
) {
    let f = c.rope.get_function("rope_uniform_pos").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(gx(rope_dim / 2, 32), n_rows as u32), 32, 0, stream>>>(
                x, cos, sin, n_rows as i32, dim as i32, rope_dim as i32, pos as i32, conj as i32
            )
        )
        .unwrap()
    }
}

/// `rope_uniform_pos_buf` (rope.cu): per-row positions (M-batch decode).
pub fn rope_uniform_pos_buf(
    c: &Cuda,
    x: F32,
    cos: F32,
    sin: F32,
    pos_per_row: I32,
    n_rows: usize,
    dim: usize,
    rope_dim: usize,
    conj: bool,
) {
    let f = c.rope.get_function("rope_uniform_pos_buf").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(gx(rope_dim / 2, 32), n_rows as u32), 32, 0, stream>>>(
                x, cos, sin, pos_per_row, n_rows as i32, dim as i32, rope_dim as i32, conj as i32
            )
        )
        .unwrap()
    }
}

/// `act_quant_ue8m0_snap` (actquant.cu): in-place fp8 snap, f32 in/out.
pub fn snap8(c: &Cuda, x: F32, n: usize) {
    let f = c.actquant.get_function("act_quant_ue8m0_snap").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<gx(n, 32), 32, 0, stream>>>(x, n as i32)).unwrap() }
}

/// `act_quant_fp8_emit` (actquant.cu): f32 -> e4m3 bytes + ue8m0 scale
/// bytes — the quantized activation storage the gemms consume.
pub fn emit8(c: &Cuda, x: F32, q: U8, s: U8, n: usize) {
    let f = c.actquant.get_function("act_quant_fp8_emit").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<gx(n, 32), 32, 0, stream>>>(x, q, s, n as i32)).unwrap() }
}

pub fn pack_scales(c: &Cuda, src: U8, dst: U32, n: usize) {
    let f = c.actquant.get_function("pack_scale_words").unwrap();
    let stream = &c.stream;
    let words = n / 128;
    unsafe { launch!(f<<<gx(words, 256), 256, 0, stream>>>(src, dst, n as i32)).unwrap() }
}

#[derive(Clone, Copy)]
pub enum TlGemm {
    Qa,
    Qb,
    Wkv,
    WoA,
    WoB,
    SharedW13,
    SharedW2,
    EngramWkv,
    DrafterMainProj,
}

/// Production TileLang block-scaled tensor-core decode GEMM (M <= 16).
/// `sa` points at the packed-word half of an activation scale buffer;
/// `sb` is the weight's upload-time packed scale table.
pub fn fp8_gemm_tl(c: &Cuda, kind: TlGemm, a: U8, sa: U32, b: U8, sb: U32, out: F32, m: usize) {
    let (module, grid_x, shared): (_, u32, u32) = match kind {
        TlGemm::Qa => (&c.tl_q_a, 20, 31_680),
        TlGemm::Qb => (&c.tl_q_b, 64, 57_024),
        TlGemm::Wkv => (&c.tl_wkv, 8, 31_680),
        TlGemm::WoA => (&c.tl_wo_a, 32, 42_240),
        TlGemm::WoB => (&c.tl_wo_b, 40, 57_024),
        TlGemm::SharedW13 => (&c.tl_shared_w13, 36, 31_680),
        TlGemm::SharedW2 => (&c.tl_shared_w2, 80, 31_680),
        TlGemm::EngramWkv => (&c.tl_engram_wkv, 400, 31_680),
        TlGemm::DrafterMainProj => (&c.tl_drafter_main_proj, 20, 31_680),
    };
    let f = module.get_function("gemm_mxfp8_kernel").unwrap();
    unsafe {
        let status = cust::sys::cuFuncSetAttribute(
            f.to_raw(),
            cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared as i32,
        );
        assert_eq!(status, cust::sys::CUresult::CUDA_SUCCESS);
        let stream = &c.stream;
        launch!(f<<<(grid_x, (m + 15) as u32 / 16), 128, shared, stream>>>(
            a, b, out, sa, sb, m as i32, m as i32
        ))
        .unwrap()
    }
}

#[repr(C, align(128))]
#[derive(Clone, Copy)]
pub struct TensorMap([u64; 16]);
unsafe impl cust::memory::DeviceCopy for TensorMap {}

extern "C" {
    fn cuTensorMapEncodeTiled(
        map: *mut TensorMap,
        data_type: u32,
        rank: u32,
        address: *mut std::ffi::c_void,
        global_dim: *const u64,
        global_strides: *const u64,
        box_dim: *const u32,
        element_strides: *const u32,
        interleave: u32,
        swizzle: u32,
        l2_promotion: u32,
        oob_fill: u32,
    ) -> i32;
}

pub(crate) fn tensor_map_fp4(weights: U8, tiles: usize) -> TensorMap {
    let mut map = TensorMap([0; 16]);
    let dims = [128u64, 128, tiles as u64];
    let strides = [1u64, 64, 8192];
    let box_dims = [128u32, 128, 1];
    let element_strides = [1u32, 1, 1];
    unsafe {
        let status = cuTensorMapEncodeTiled(
            &mut map,
            14, // CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
            3,
            weights.as_raw() as *mut std::ffi::c_void,
            dims.as_ptr(),
            strides.as_ptr().add(1),
            box_dims.as_ptr(),
            element_strides.as_ptr(),
            0,
            3, // CU_TENSOR_MAP_SWIZZLE_128B
            2, // CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            0,
        );
        assert_eq!(status, 0);
    }
    map
}

#[allow(clippy::too_many_arguments)]
pub fn moe_dispatch_tl(
    c: &Cuda,
    topk_idx: I64,
    topk_weights: F32,
    order: I32,
    layout_weights: F32,
    route_pos: I32,
    tile_expert: I32,
    expert_count: I32,
    expert_offset: I32,
    expert_psum: I32,
    num_tiles: I32,
    rows: usize,
    top_k: usize,
    tiles: usize,
) {
    let f = c
        .tl_moe_dispatch
        .get_function("moe_dispatch_kernel")
        .unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<1, 512, 6528, stream>>>(
            expert_count,
            expert_offset,
            layout_weights,
            num_tiles,
            order,
            expert_psum,
            route_pos,
            tile_expert,
            topk_idx,
            topk_weights,
            rows as i32,
            top_k as i32,
            tiles as i32
        ))
        .unwrap()
    }
}

#[derive(Clone, Copy)]
pub enum TlMoe {
    GateUp,
    Down,
}

#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_tl(
    c: &Cuda,
    kind: TlMoe,
    a: U8,
    sfa: U32,
    weights: U8,
    sfb: U32,
    order: I32,
    tile_expert: I32,
    out: F32,
    tiles: usize,
    tokens: usize,
    top_k: usize,
) {
    let map = tensor_map_fp4(weights, 69120);
    moe_grouped_tl_cached(
        c,
        kind,
        a,
        sfa,
        map,
        sfb,
        order,
        tile_expert,
        out,
        tiles,
        tokens,
        top_k,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_tl_cached(
    c: &Cuda,
    kind: TlMoe,
    a: U8,
    sfa: U32,
    map: TensorMap,
    sfb: U32,
    order: I32,
    tile_expert: I32,
    out: F32,
    tiles: usize,
    tokens: usize,
    top_k: usize,
) {
    let (module, n_tiles) = match kind {
        TlMoe::GateUp => (&c.tl_moe_gate_up, 18),
        TlMoe::Down => (&c.tl_moe_down, 40),
    };
    // Gate/up reads token rows; down reads the grouped SwiGLU layout.
    let input_rows = match kind {
        TlMoe::GateUp => tokens,
        TlMoe::Down => tiles * 32,
    };
    let f = module
        .get_function("moe_grouped_fp8_fp4_gemm_kernel")
        .unwrap();
    unsafe {
        let status = cust::sys::cuFuncSetAttribute(
            f.to_raw(),
            cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            64000,
        );
        assert_eq!(status, cust::sys::CUresult::CUDA_SUCCESS);
        let stream = &c.stream;
        match kind {
            TlMoe::GateUp => launch!(f<<<(n_tiles, tiles as u32), 128, 64000, stream>>>(
                a, map, out, order, sfa, sfb, tile_expert, tiles as i32, input_rows as i32, top_k as i32
            )),
            TlMoe::Down => launch!(f<<<(n_tiles, tiles as u32), 128, 64000, stream>>>(
                a, map, out, order, sfa, sfb, tile_expert, tiles as i32, input_rows as i32
            )),
        }
        .unwrap()
    }
}

pub fn fp4_emit(c: &Cuda, x: F32, packed: U8, scales: U8, n: usize, block: usize) {
    let f = c.fp4_actquant.get_function("fp4_act_quant_emit").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(n / block) as u32, 32, 0, stream>>>(x, packed, scales, block as i32)).unwrap()
    }
}

pub fn f32_to_bf16(c: &Cuda, x: F32, y: U16, n: usize) {
    let f = c.attn.get_function("f32_to_bf16").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<gx(n, 256), 256, 0, stream>>>(x, y, n as i32)).unwrap() }
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_mla_decode(
    c: &Cuda,
    q: U16,
    swa_records: U8,
    swa_idxs: I32,
    swa_len: I32,
    idx_records: U8,
    idx_idxs: I32,
    idx_len: I32,
    page_table: I32,
    sinks: F32,
    positions: I64,
    cos: F32,
    sin: F32,
    partial: F32,
    stats: F32,
    out: F32,
    n_heads: usize,
    idx_pages: usize,
    idx_pool: usize,
    idx_stride: usize,
    swa_pool: usize,
    swa_stride: usize,
    positions_max: usize,
    segments: usize,
) {
    let split = c.tl_sparse_mla.get_function("sparse_mla_kernel").unwrap();
    let merge = c.tl_sparse_mla.get_function("sparse_mla_kernel_1").unwrap();
    unsafe {
        let status = cust::sys::cuFuncSetAttribute(
            split.to_raw(),
            cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            101376,
        );
        assert_eq!(status, cust::sys::CUresult::CUDA_SUCCESS);
        let stream = &c.stream;
        launch!(split<<<(segments as u32, 1, 1), 256, 101376, stream>>>(
            idx_idxs, idx_len, page_table, idx_records, partial, stats, q,
            swa_idxs, swa_len, swa_records,
            idx_pages as i32, idx_pool as i32, idx_stride as i32,
            1, swa_pool as i32, swa_stride as i32
        ))
        .unwrap();
        launch!(merge<<<(n_heads as u32, 1), 128, 0, stream>>>(
            cos, sin, out, partial, stats, positions, sinks,
            positions_max as i32, 1, segments as i32
        ))
        .unwrap();
    }
}

pub fn moe_dispatch_ids(
    c: &Cuda,
    gate_idx: I32,
    dispatch_idx: I64,
    top_k: usize,
    e_start: usize,
    e_len: usize,
) {
    let f = c.moe_grouped.get_function("moe_dispatch_ids").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<1, 32, 0, stream>>>(gate_idx, dispatch_idx, top_k as i32, e_start as i32, e_len as i32)).unwrap()
    }
}

pub fn moe_swiglu_route(
    c: &Cuda,
    gate: F32,
    up: F32,
    weights: F32,
    out: F32,
    inter: usize,
    rows: usize,
) {
    let f = c.moe_grouped.get_function("moe_swiglu_route").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<((inter / 256) as u32, rows as u32), 256, 0, stream>>>(gate, up, weights, out, inter as i32)).unwrap()
    }
}

pub fn moe_combine_f32(
    c: &Cuda,
    y: F32,
    route_pos: I32,
    out: F32,
    dim: usize,
    top_k: usize,
    rows: usize,
) {
    let f = c.moe_grouped.get_function("moe_combine_f32").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(gx(dim, 256), rows as u32), 256, 0, stream>>>(y, route_pos, out, dim as i32, top_k as i32)).unwrap()
    }
}

/// `fp4_act_quant_snap` (fp4_actquant.cu): e8m0 path (indexer, block 32)
/// or e4m3 path (compressed KV, block 16).
pub fn fp4_snap(c: &Cuda, x: F32, n: usize, block: usize, scale_e4m3: bool) {
    let f = c.fp4_actquant.get_function("fp4_act_quant_snap").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<gx(n, block as u32), 32, 0, stream>>>(x, n as i32, block as i32, scale_e4m3 as i32)).unwrap()
    }
}

/// `fp8_gemm` (fp8_gemm.cu): C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm(c: &Cuda, a: U8, sa: U8, b: U8, sb: U8, out: F32, m: usize, n: usize, k: usize) {
    let f = c.fp8_gemm.get_function("fp8_gemm").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(n as u32, m as u32), 256, 0, stream>>>(a, sa, b, sb, out, m as i32, n as i32, k as i32)).unwrap()
    }
}

/// `fp8_gemm_tile` (fp8_gemm_tile.cu): M-batch tiled form of fp8_gemm —
/// one 16x32 output tile per block, each thread owning one output
/// accumulated over whole K-blocks (no block_reduce). For M > 1 batched
/// steps; the M=1 path keeps fp8_gemm (bit-exact with the gate history).
pub fn fp8_gemm_tile(
    c: &Cuda,
    a: U8,
    sa: U8,
    b: U8,
    sb: U8,
    out: F32,
    m: usize,
    n: usize,
    k: usize,
) {
    let f = c.fp8_gemm_tile.get_function("fp8_gemm_tile").unwrap();
    let stream = &c.stream;
    let gx = n.div_ceil(32) as u32;
    let gy = m.div_ceil(16) as u32;
    unsafe {
        launch!(
            f<<<(gx, gy), 512, 0, stream>>>(a, sa, b, sb, out, m as i32, n as i32, k as i32)
        )
        .unwrap()
    }
}

/// `fp4_expert_gemm` (fp4_expert_gemm.cu): fp8 act × packed fp4 weight.
#[allow(clippy::too_many_arguments)]
pub fn fp4_gemm(c: &Cuda, a: U8, sa: U8, b: U8, sb: U8, out: F32, m: usize, n: usize, k: usize) {
    let f = c.fp4_expert_gemm.get_function("fp4_expert_gemm").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(n as u32, m as u32), 256, 0, stream>>>(a, sa, b, sb, out, m as i32, n as i32, k as i32)).unwrap()
    }
}

/// `fp4_gemm_tile` (fp4_gemm_tile.cu): M-batch tiled form of the fp4
/// expert GEMM — one 16x32 output tile per block, whole-K per thread.
/// For M > 1 batched steps (DSpark verify); M=1 keeps the fused
/// grouped-expert path.
pub fn fp4_gemm_tile(
    c: &Cuda,
    a: U8,
    sa: U8,
    b: U8,
    sb: U8,
    out: F32,
    m: usize,
    n: usize,
    k: usize,
) {
    let f = c.fp4_gemm_tile.get_function("fp4_gemm_tile").unwrap();
    let stream = &c.stream;
    let gx = n.div_ceil(32) as u32;
    let gy = m.div_ceil(16) as u32;
    unsafe {
        launch!(
            f<<<(gx, gy), 512, 0, stream>>>(a, sa, b, sb, out, m as i32, n as i32, k as i32)
        )
        .unwrap()
    }
}

/// `hc_mixes` (hc.cu): one warp per dot row, grid (mix_hc, n_tokens) —
/// the dots spread across SMs instead of 8 warps of one block.
pub fn hc_mixes(
    c: &Cuda,
    xf: F32,
    fnc: F32,
    dots: F32,
    n_tokens: usize,
    hc_dim: usize,
    mix_hc: usize,
) {
    let f = c.hc.get_function("hc_mixes").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(mix_hc as u32, n_tokens as u32), 32, 0, stream>>>(
                xf, fnc, dots, n_tokens as i32, hc_dim as i32, mix_hc as i32
            )
        )
        .unwrap()
    }
}

/// `hc_finalize` (hc.cu): rstd (original block_reduce_sum shape) + the
/// sigmoid/softmax/Sinkhorn tail over hc_mixes's dots.
#[allow(clippy::too_many_arguments)]
pub fn hc_finalize(
    c: &Cuda,
    xf: F32,
    dots: F32,
    base: F32,
    scale: F32,
    pre: F32,
    post: F32,
    comb: F32,
    n_tokens: usize,
    hc_dim: usize,
    mix_hc: usize,
    eps: f32,
) {
    let f = c.hc.get_function("hc_finalize").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<n_tokens as u32, 256, 0, stream>>>(
                xf, dots, base, scale, pre, post, comb,
                n_tokens as i32, hc_dim as i32, mix_hc as i32, eps
            )
        )
        .unwrap()
    }
}

/// `hc_block` (hc.cu): mixes + 20-iter Sinkhorn for `n_tokens` tokens.
#[allow(clippy::too_many_arguments)]
pub fn hc_block(
    c: &Cuda,
    xf: F32,
    fnc: F32,
    base: F32,
    scale: F32,
    pre: F32,
    post: F32,
    comb: F32,
    n_tokens: usize,
    hc_dim: usize,
    mix_hc: usize,
    eps: f32,
) {
    let f = c.hc.get_function("hc_block").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<n_tokens as u32, 256, 0, stream>>>(
                xf, fnc, base, scale, pre, post, comb, n_tokens as i32, hc_dim as i32, mix_hc as i32, eps
            )
        )
        .unwrap()
    }
}

/// `hc_pre` (hc.cu): collapse the hc copies weighted by `pre`.
pub fn hc_pre(c: &Cuda, x: F32, pre: F32, y: F32, n_tokens: usize, dim: usize) {
    let f = c.hc.get_function("hc_pre").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(gx(dim, 128), n_tokens as u32), 128, 0, stream>>>(x, pre, y, n_tokens as i32, dim as i32)).unwrap()
    }
}

/// `hc_post` (hc.cu): expand + Sinkhorn-mix the residual.
pub fn hc_post(
    c: &Cuda,
    x: F32,
    res: F32,
    post: F32,
    comb: F32,
    y: F32,
    n_tokens: usize,
    dim: usize,
) {
    let f = c.hc.get_function("hc_post").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(gx(dim, 128), n_tokens as u32), 128, 0, stream>>>(x, res, post, comb, y, n_tokens as i32, dim as i32)).unwrap()
    }
}

/// `sparse_attn` (attn.cu): one block per head over gathered kv rows.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn(
    c: &Cuda,
    q: F32,
    kv: F32,
    idxs: I32,
    sink: F32,
    out: F32,
    n_heads: usize,
    dim: usize,
    n_sel: usize,
    scale: f32,
) {
    let f = c.attn.get_function("sparse_attn").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<n_heads as u32, 128, 0, stream>>>(
                q, kv, idxs, sink, out, n_heads as i32, dim as i32, n_sel as i32, scale
            )
        )
        .unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_split(
    c: &Cuda,
    q: F32,
    kv: F32,
    idxs: I32,
    sink: F32,
    partial: F32,
    stats: F32,
    out: F32,
    n_heads: usize,
    dim: usize,
    n_sel: usize,
    scale: f32,
) {
    let segments = n_sel.div_ceil(128);
    let split = c.attn.get_function("sparse_attn_split").unwrap();
    let merge = c.attn.get_function("sparse_attn_merge").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(split<<<(n_heads as u32, segments as u32), 128, 0, stream>>>(
            q, kv, idxs, partial, stats, n_heads as i32, dim as i32, n_sel as i32, scale
        ))
        .unwrap();
        launch!(merge<<<n_heads as u32, 128, 0, stream>>>(
            partial, stats, sink, out, n_heads as i32, dim as i32, segments as i32
        ))
        .unwrap();
    }
}

/// `sparse_attn_m` (attn.cu): M-batch sparse attention — grid (n_heads, m),
/// bit-identical to m sequential sparse_attn calls per (head, position).
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_m(
    c: &Cuda,
    q: F32,
    kv: F32,
    idxs: I32,
    sink: F32,
    out: F32,
    n_heads: usize,
    dim: usize,
    n_sel: usize,
    m: usize,
    scale: f32,
) {
    let f = c.attn.get_function("sparse_attn_m").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(n_heads as u32, m as u32), 128, 0, stream>>>(
                q, kv, idxs, sink, out, n_heads as i32, dim as i32, n_sel as i32, scale
            )
        )
        .unwrap()
    }
}

/// `moe_gate` (moe.cu): per-expert parallel gate scoring, one block/expert.
pub fn moe_gate(
    c: &Cuda,
    x: F32,
    gate_w: U16,
    bias: F32,
    scores: F32,
    dim: usize,
    n_experts: usize,
) {
    let f = c.moe.get_function("moe_gate").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<n_experts as u32, 256, 0, stream>>>(
                x, gate_w, bias, scores, dim as i32, n_experts as i32
            )
        )
        .unwrap()
    }
}

/// `moe_gate_m` (moe.cu): M-batch gate scoring — grid (n_experts, m).
pub fn moe_gate_m(
    c: &Cuda,
    x: F32,
    gate_w: U16,
    bias: F32,
    scores: F32,
    m: usize,
    dim: usize,
    n_experts: usize,
) {
    let f = c.moe.get_function("moe_gate_m").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(n_experts as u32, m as u32), 256, 0, stream>>>(
                x, gate_w, bias, scores, dim as i32, n_experts as i32
            )
        )
        .unwrap()
    }
}

/// `moe_select` (moe.cu): bias-steered top-k over the gate scores.
pub fn moe_select(
    c: &Cuda,
    scores: F32,
    bias: F32,
    out_w: F32,
    out_i: I32,
    n_experts: usize,
    top_k: usize,
    route_scale: f32,
) {
    let f = c.moe.get_function("moe_select").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<1, 32, 0, stream>>>(
                scores, bias, out_w, out_i, n_experts as i32, top_k as i32,
                route_scale
            )
        )
        .unwrap()
    }
}

/// `moe_route` (moe.cu): sqrtsoftplus gate + bias-steered top-k.
#[allow(clippy::too_many_arguments)]
pub fn moe_route(
    c: &Cuda,
    x: F32,
    gate_w: U16,
    bias: F32,
    out_w: F32,
    out_i: I32,
    dim: usize,
    n_experts: usize,
    top_k: usize,
    route_scale: f32,
) {
    let f = c.moe.get_function("moe_route").unwrap();
    let stream = &c.stream;
    let block = n_experts.min(256) as u32;
    let shared = (n_experts * 4) as u32;
    unsafe {
        launch!(
            f<<<1u32, block, shared, stream>>>(
                x, gate_w, bias, out_w, out_i, dim as i32, n_experts as i32, top_k as i32, route_scale
            )
        )
        .unwrap()
    }
}

/// `swiglu_limit10` (moe.cu): asymmetric clamps, silu, routing weight.
pub fn swiglu(c: &Cuda, gate: F32, up: F32, out: F32, n: usize, rw: f32) {
    let f = c.moe.get_function("swiglu_limit10").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<gx(n, 256), 256, 0, stream>>>(gate, up, out, n as i32, rw)).unwrap() }
}

/// `compressor_finish` (compressor.cu): per-dim softmax pooling of the two
/// parked slots. (The bf16 snap that follows it in the kernel file is NOT
/// applied — the CPU reference keeps the latent in f32; see crate docs.)
pub fn compressor_finish(c: &Cuda, kv_state: F32, score_state: F32, latent: F32, dim: usize) {
    let f = c.compressor.get_function("compressor_finish").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<gx(dim, 256), 256, 0, stream>>>(kv_state, score_state, latent, dim as i32))
            .unwrap()
    }
}

/// `indexer_score` (indexer.cu): relu q·k dots combined per-head.
#[allow(clippy::too_many_arguments)]
pub fn indexer_score(
    c: &Cuda,
    q: F32,
    index_k: F32,
    weights: F32,
    out: F32,
    n_heads: usize,
    head_dim: usize,
    n_rows: usize,
) {
    let f = c.indexer.get_function("indexer_score").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<n_rows as u32, 32, 0, stream>>>(
                q, index_k, weights, out, n_heads as i32, head_dim as i32, n_rows as i32
            )
        )
        .unwrap()
    }
}

/// `f32_gemv` (gemv.cu).
pub fn f32_gemv(c: &Cuda, w: F32, x: F32, y: F32, rows: usize, cols: usize) {
    let f = c.gemv.get_function("f32_gemv").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<rows as u32, 256, 0, stream>>>(w, x, y, rows as i32, cols as i32)).unwrap()
    }
}

/// `bf16_gemv` (gemv.cu).
pub fn bf16_gemv(c: &Cuda, w: U16, x: F32, y: F32, rows: usize, cols: usize) {
    let f = c.gemv.get_function("bf16_gemv").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<rows as u32, 256, 0, stream>>>(w, x, y, rows as i32, cols as i32)).unwrap()
    }
}

/// `acc_f32` (gemv.cu): y += x (the MoE accumulator).
pub fn acc_f32(c: &Cuda, x: F32, y: F32, n: usize) {
    let f = c.gemv.get_function("acc_f32").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<gx(n, 256), 256, 0, stream>>>(x, y, n as i32)).unwrap() }
}

/// `gather_rows_fp8` (engram.cu): fp8 table row gather with ue8m0 scales.
#[allow(clippy::too_many_arguments)]
pub fn gather_fp8(c: &Cuda, table: U8, scales: U8, idxs: I64, out: F32, n_sel: usize, cols: usize) {
    let f = c.engram.get_function("gather_rows_fp8").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<n_sel as u32, 256, 0, stream>>>(table, scales, idxs, out, n_sel as i32, cols as i32)).unwrap()
    }
}

/// `gather_rows_f32` (engram.cu): f32 table row gather (embedding lookup).
pub fn gather_f32(c: &Cuda, table: F32, idxs: I64, out: F32, n_sel: usize, cols: usize) {
    let f = c.engram.get_function("gather_rows_f32").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<n_sel as u32, 256, 0, stream>>>(table, idxs, out, n_sel as i32, cols as i32))
            .unwrap()
    }
}

/// `engram_merge` (engram.cu): gated merge for ONE hc copy (the kernel
/// offsets `h` by `blockIdx.x`; the engine passes the per-copy slice so
/// each copy gets its own `w` row).
pub fn engram_merge(c: &Cuda, h: F32, key: F32, value: F32, w: F32, dim: usize, inv_sqrt_dim: f32) {
    let f = c.engram.get_function("engram_merge").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<1u32, 256, 0, stream>>>(h, key, value, w, dim as i32, inv_sqrt_dim)).unwrap()
    }
}

/// `argmax_rows` (sample.cu): greedy pick, ties to the lowest index.
pub fn argmax(c: &Cuda, logits: F32, out_i: I32, out_v: F32, n: usize) {
    let f = c.sample.get_function("argmax_rows").unwrap();
    let stream = &c.stream;
    unsafe { launch!(f<<<1u32, 256, 0, stream>>>(logits, out_i, out_v, n as i32)).unwrap() }
}

/// Stream-ordered device-to-device copy of `n` f32 elements.
pub fn copy_f32(c: &Cuda, src: F32, dst: F32, n: usize) {
    let src = unsafe { DeviceSlice::from_raw_parts(src, n) };
    let mut dst = unsafe { DeviceSlice::from_raw_parts_mut(dst, n) };
    unsafe { dst.async_copy_from(&src, &c.stream).unwrap() }
}
