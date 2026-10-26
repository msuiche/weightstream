//! Driver context, device probe and kernel module loading.

use cust::context::CurrentContext;
use cust::prelude::*;
use std::sync::Mutex;

const DEQUANT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/dequant.ptx"));
const NORM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/norm.ptx"));
const ROPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));
const ACTQUANT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/actquant.ptx"));
const FP8_GEMM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fp8_gemm.ptx"));
const FP4_ACTQUANT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fp4_actquant.ptx"));
const FP4_EXPERT_GEMM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fp4_expert_gemm.ptx"));
const HC_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hc.ptx"));
const ATTN_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attn.ptx"));
const MOE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/moe.ptx"));
const GEMV_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gemv.ptx"));
const COMPRESSOR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/compressor.ptx"));
const INDEXER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/indexer.ptx"));
const ENGRAM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/engram.ptx"));
const SAMPLE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sample.ptx"));

pub struct Cuda {
    pub context: Context,
    pub device: Device,
    pub stream: Stream,
    pub dequant: Module,
    pub norm: Module,
    pub rope: Module,
    pub actquant: Module,
    pub fp8_gemm: Module,
    pub fp4_actquant: Module,
    pub fp4_expert_gemm: Module,
    pub hc: Module,
    pub attn: Module,
    pub moe: Module,
    pub gemv: Module,
    pub compressor: Module,
    pub indexer: Module,
    pub engram: Module,
    pub sample: Module,
}

static CUDA: Mutex<Option<Cuda>> = Mutex::new(None);

#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub sm_count: u32,
    pub memory_bytes: u64,
}

impl Cuda {
    /// One global context for the process; cargo tests run multithreaded,
    /// so every public entry point takes the lock and re-sets current.
    pub fn global() -> std::sync::MutexGuard<'static, Option<Cuda>> {
        let mut guard = CUDA.lock().unwrap();
        if guard.is_none() {
            let context = cust::quick_init().expect("cuda init");
            let device = Device::get_device(0).expect("device 0");
            let stream = Stream::new(StreamFlags::DEFAULT, None).expect("stream");
            let dequant = Module::from_ptx(DEQUANT_PTX, &[]).expect("dequant module");
            let norm = Module::from_ptx(NORM_PTX, &[]).expect("norm module");
            let rope = Module::from_ptx(ROPE_PTX, &[]).expect("rope module");
            let actquant = Module::from_ptx(ACTQUANT_PTX, &[]).expect("actquant module");
            let fp8_gemm = Module::from_ptx(FP8_GEMM_PTX, &[]).expect("fp8_gemm module");
            let fp4_actquant =
                Module::from_ptx(FP4_ACTQUANT_PTX, &[]).expect("fp4_actquant module");
            let fp4_expert_gemm =
                Module::from_ptx(FP4_EXPERT_GEMM_PTX, &[]).expect("fp4_expert_gemm module");
            let hc = Module::from_ptx(HC_PTX, &[]).expect("hc module");
            let attn = Module::from_ptx(ATTN_PTX, &[]).expect("attn module");
            let moe = Module::from_ptx(MOE_PTX, &[]).expect("moe module");
            let gemv = Module::from_ptx(GEMV_PTX, &[]).expect("gemv module");
            let compressor = Module::from_ptx(COMPRESSOR_PTX, &[]).expect("compressor module");
            let indexer = Module::from_ptx(INDEXER_PTX, &[]).expect("indexer module");
            let engram = Module::from_ptx(ENGRAM_PTX, &[]).expect("engram module");
            let sample = Module::from_ptx(SAMPLE_PTX, &[]).expect("sample module");
            *guard = Some(Cuda {
                context,
                device,
                stream,
                dequant,
                norm,
                rope,
                actquant,
                fp8_gemm,
                fp4_actquant,
                fp4_expert_gemm,
                hc,
                attn,
                moe,
                gemv,
                compressor,
                indexer,
                engram,
                sample,
            });
        }
        // tests run on many threads; the lock holder always becomes current
        CurrentContext::set_current(&guard.as_ref().unwrap().context).expect("set current");
        guard
    }

    pub fn info(device: Device) -> DeviceInfo {
        DeviceInfo {
            name: device.name().unwrap_or_default(),
            sm_count: device
                .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)
                .unwrap_or(0) as u32,
            memory_bytes: device.total_memory().unwrap_or(0) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_gb10() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let info = Cuda::info(cuda.device);
        eprintln!("device: {info:?}");
        assert_eq!(info.sm_count, 48, "GB10 has 48 SMs");
        assert!(info.memory_bytes > 100 * (1 << 30));
    }

    #[test]
    fn fp4_dequant_matches_cpu_reference() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();

        // 4 blocks of pseudo-random packed FP4 + scales
        let n_blocks = 4;
        let mut packed = Vec::with_capacity(n_blocks * 16);
        let mut scales = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks * 16 {
            packed.push((i as u8).wrapping_mul(37).wrapping_add(11));
        }
        for i in 0..n_blocks {
            scales.push(120 + 2 * i as u8); // E8M0 exponents around 1/128..1
        }

        // CPU reference
        let mut expected = vec![0.0f32; n_blocks * 32];
        for b in 0..n_blocks {
            let scale = ferroseek_core::quant::e8m0_to_f32(scales[b]);
            let mut block = [0.0f32; 32];
            ferroseek_core::quant::dequant_fp4_block(
                &packed[b * 16..(b + 1) * 16],
                scale,
                &mut block,
            );
            expected[b * 32..(b + 1) * 32].copy_from_slice(&block);
        }

        // GPU
        let d_packed = DeviceBuffer::from_slice(&packed).unwrap();
        let d_scales = DeviceBuffer::from_slice(&scales).unwrap();
        let mut d_out = DeviceBuffer::from_slice(&vec![0.0f32; n_blocks * 32]).unwrap();
        let func = cuda.dequant.get_function("fp4_dequant").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<1, 32, 0, stream>>>(
                    d_packed.as_device_ptr(),
                    d_scales.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n_blocks as i32
                )
            )
            .unwrap();
        }
        cuda.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n_blocks * 32];
        d_out.copy_to(&mut got).unwrap();

        assert_eq!(got, expected, "GPU dequant diverges from CPU reference");
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn rmsnorm_matches_cpu() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (rows, dim) = (8usize, 1280usize);
        let x = rand_vec(rows * dim, 42);
        let w = rand_vec(dim, 43);
        let eps = 1e-20f32;

        let mut expected = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let ms: f32 = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rstd = 1.0 / (ms + eps).sqrt();
            for i in 0..dim {
                expected[r * dim + i] = xr[i] * rstd * w[i];
            }
        }

        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_w = DeviceBuffer::from_slice(&w).unwrap();
        let mut d_out = DeviceBuffer::from_slice(&vec![0.0; rows * dim]).unwrap();
        let func = cuda.norm.get_function("rmsnorm").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<rows as u32, 256, 0, stream>>>(
                    d_x.as_device_ptr(),
                    d_w.as_device_ptr(),
                    d_out.as_device_ptr(),
                    dim as i32,
                    eps
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; rows * dim];
        d_out.copy_to(&mut got).unwrap();

        let max_rel: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| ((g - e).abs()) / e.abs().max(1e-6))
            .fold(0.0, f32::max);
        assert!(max_rel < 1e-5, "rmsnorm max rel err {max_rel}");
    }

    #[test]
    fn rope_matches_cpu_incl_conjugate() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (rows, dim, rope_dim) = (6usize, 512usize, 64usize);
        let pos_offset = 130_000usize; // exercise YaRN table at long context
        let table = crate::rope_tables::compress(rope_dim, pos_offset + rows);
        let mut x = rand_vec(rows * dim, 77);

        let cpu_apply = |x: &mut [f32], conj: bool| {
            for r in 0..rows {
                let pos = pos_offset + r;
                for p in 0..rope_dim / 2 {
                    let c = table.cos[pos * rope_dim / 2 + p];
                    let mut s = table.sin[pos * rope_dim / 2 + p];
                    if conj {
                        s = -s;
                    }
                    let base = r * dim + (dim - rope_dim) + 2 * p;
                    let (x0, x1) = (x[base], x[base + 1]);
                    x[base] = x0 * c - x1 * s;
                    x[base + 1] = x1 * c + x0 * s;
                }
            }
        };

        for conj in [false, true] {
            let mut expected = x.clone();
            cpu_apply(&mut expected, conj);
            let mut d_x = DeviceBuffer::from_slice(&x).unwrap();
            let d_cos = DeviceBuffer::from_slice(&table.cos).unwrap();
            let d_sin = DeviceBuffer::from_slice(&table.sin).unwrap();
            let func = cuda.rope.get_function("rope").unwrap();
            let stream = &cuda.stream;
            unsafe {
                launch!(
                    func<<<(32u32, rows as u32), 32, 0, stream>>>(
                        d_x.as_device_ptr(),
                        d_cos.as_device_ptr(),
                        d_sin.as_device_ptr(),
                        rows as i32,
                        dim as i32,
                        rope_dim as i32,
                        pos_offset as i32,
                        conj as i32
                    )
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            d_x.copy_to(&mut x).unwrap();
            let max_err: f32 = x
                .iter()
                .zip(&expected)
                .map(|(g, e)| (g - e).abs())
                .fold(0.0, f32::max);
            assert!(max_err < 1e-5, "rope conj={conj} max err {max_err}");
        }
    }

    fn frexp(v: f32) -> (f32, i32) {
        if v == 0.0 {
            return (0.0, 0);
        }
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32 - 126;
        let m = f32::from_bits((bits & 0x807F_FFFF) | 0x3F00_0000);
        (m, exp)
    }

    /// CPU mirror of the CUDA f32→e4m3 (round-to-nearest-even, saturate 448)
    fn f32_to_e4m3(v: f32) -> f32 {
        let sign = v.is_sign_negative();
        let a = v.abs().min(448.0);
        if a == 0.0 {
            return if sign { -0.0 } else { 0.0 };
        }
        let (m, exp) = frexp(a);
        let mut e = exp - 1 + 7;
        let val = if e <= 0 {
            (a * 512.0).round() / 512.0
        } else {
            let mut mant = ((m * 2.0 - 1.0) * 8.0).round() as i32;
            if mant == 8 {
                mant = 0;
                e += 1;
            }
            // E=15 mant 0..6 is the valid 256..448 range; only e>=16
            // saturates (448 = S.1111.110)
            if e >= 16 {
                448.0
            } else {
                (1.0 + mant as f32 / 8.0) * 2f32.powi(e - 7)
            }
        };
        if sign {
            -val
        } else {
            val
        }
    }

    #[test]
    fn act_quant_snap_bit_exact() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let n = 256usize; // 8 groups of 32
        let mut x = rand_vec(n, 99);
        // include extremes: tiny group (amax floor) and huge values (clamp)
        for v in x.iter_mut().take(32) {
            *v *= 1e-6;
        }
        x[40] = 900.0;
        x[41] = -1200.0;

        let mut expected = x.clone();
        for g in expected.chunks_exact_mut(32) {
            let amax = g.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-4);
            let s = 2f32.powf((amax / 448.0).log2().ceil());
            for v in g.iter_mut() {
                let y = (*v / s).clamp(-448.0, 448.0);
                *v = f32_to_e4m3(y) * s;
            }
        }

        let mut d_x = DeviceBuffer::from_slice(&x).unwrap();
        let func = cuda.actquant.get_function("act_quant_ue8m0_snap").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(func<<<8u32, 32, 0, stream>>>(d_x.as_device_ptr(), n as i32)).unwrap();
        }
        stream.synchronize().unwrap();
        d_x.copy_to(&mut x).unwrap();
        assert_eq!(x, expected, "act_quant snap not bit-exact");
    }

    /// Quantize f32 → (e4m3 bytes, e8m0 scales) per 32-block, mirroring
    /// act_quant_ue8m0_snap's scale rule but emitting quantized storage.
    fn quantize_fp8(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let kb = x.len() / 32;
        let mut q = vec![0u8; x.len()];
        let mut s = vec![0u8; kb];
        for (b, g) in x.chunks_exact(32).enumerate() {
            let amax = g.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-4);
            let scale = 2f32.powf((amax / 448.0).log2().ceil());
            s[b] = (scale.log2() + 127.0) as u8;
            for (i, &v) in g.iter().enumerate() {
                let y = (v / scale).clamp(-448.0, 448.0);
                q[b * 32 + i] = f32_to_e4m3_bits(y);
            }
        }
        (q, s)
    }

    /// f32 → e4m3 bits (RNE, saturate 448) — bit-level mirror used for
    /// building kernel inputs.
    fn f32_to_e4m3_bits(v: f32) -> u8 {
        let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
        let a = v.abs().min(448.0);
        if a == 0.0 {
            return sign;
        }
        let (m, exp) = frexp(a);
        let mut e = exp - 1 + 7;
        if e <= 0 {
            // subnormal: quantum 2^-9
            let mant = (a / 2f32.powi(-9)).round() as u8;
            return sign | mant.min(7);
        }
        let mut mant = ((m * 2.0 - 1.0) * 8.0).round() as i32;
        if mant == 8 {
            mant = 0;
            e += 1;
        }
        // E=15 mant 0..6 is the valid 256..448 range; only e>=16 saturates
        if e >= 16 {
            return sign | 0x7E; // saturate to 448 (S.1111.110)
        }
        sign | ((e as u8) << 3) | (mant as u8)
    }

    #[test]
    fn fp8_gemm_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (m_dim, n_dim, k_dim) = (3usize, 96usize, 8192usize);
        let kb = k_dim / 32;

        let a_f32 = rand_vec(m_dim * k_dim, 7);
        let b_f32 = rand_vec(n_dim * k_dim, 8);
        let (a_q, a_s) = quantize_fp8(&a_f32);
        let (b_q, b_s) = quantize_fp8(&b_f32);

        // CPU mirror: same block structure and scale rule, f32 accumulate
        let mut expected = vec![0.0f32; m_dim * n_dim];
        for m in 0..m_dim {
            for n in 0..n_dim {
                let mut acc = 0.0f32;
                for b in 0..kb {
                    let mut dot = 0.0f32;
                    for k in 0..32 {
                        let av = ferroseek_core::quant::e4m3_to_f32(a_q[m * k_dim + b * 32 + k]);
                        let bv = ferroseek_core::quant::e4m3_to_f32(b_q[n * k_dim + b * 32 + k]);
                        dot += av * bv;
                    }
                    let sa = ferroseek_core::quant::e8m0_to_f32(a_s[m * kb + b]);
                    let sb = ferroseek_core::quant::e8m0_to_f32(b_s[(n / 32) * kb + b]);
                    acc += dot * sa * sb;
                }
                expected[m * n_dim + n] = acc;
            }
        }

        let d_a = DeviceBuffer::from_slice(&a_q).unwrap();
        let d_as = DeviceBuffer::from_slice(&a_s).unwrap();
        let d_b = DeviceBuffer::from_slice(&b_q).unwrap();
        let d_bs = DeviceBuffer::from_slice(&b_s).unwrap();
        let mut d_c = DeviceBuffer::from_slice(&vec![0.0f32; m_dim * n_dim]).unwrap();
        let func = cuda.fp8_gemm.get_function("fp8_gemm").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<(n_dim as u32, m_dim as u32), 256, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_as.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_bs.as_device_ptr(),
                    d_c.as_device_ptr(),
                    m_dim as i32,
                    n_dim as i32,
                    k_dim as i32
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; m_dim * n_dim];
        d_c.copy_to(&mut got).unwrap();

        let max_rel: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs() / e.abs().max(1.0))
            .fold(0.0, f32::max);
        assert!(max_rel < 1e-4, "fp8_gemm max rel err {max_rel}");
    }

    /// f32 → nearest e2m1 value, RNE with ties to even mantissa (CPU mirror
    /// of __nv_cvt_float_to_fp4).
    fn f32_to_fp4_cpu(v: f32) -> f32 {
        const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let sign = v.is_sign_negative();
        let a = v.abs();
        // nearest grid point; ties to even code index
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for (i, &g) in GRID.iter().enumerate() {
            let d = (a - g).abs();
            if d < best_d - 1e-12 || (d <= best_d + 1e-12 && d == best_d && i % 2 == 0) {
                if d < best_d || i % 2 == 0 {
                    best = i;
                    best_d = d;
                }
            }
        }
        let r = GRID[best];
        if sign {
            -r
        } else {
            r
        }
    }

    #[test]
    fn fp4_encode_rne_ties() {
        // exact midpoints round to even-mantissa neighbor
        assert_eq!(f32_to_fp4_cpu(0.25), 0.0); // 0 (code 0) vs 0.5 (code 1)
        assert_eq!(f32_to_fp4_cpu(0.75), 1.0); // 0.5 (code 1) vs 1.0 (code 2)
        assert_eq!(f32_to_fp4_cpu(1.25), 1.0); // 1.0 (code 2) vs 1.5 (code 3)
        assert_eq!(f32_to_fp4_cpu(1.75), 2.0); // 1.5 (3) vs 2.0 (4)
        assert_eq!(f32_to_fp4_cpu(2.5), 2.0); // 2.0 (4) vs 3.0 (5)
        assert_eq!(f32_to_fp4_cpu(3.5), 4.0); // 3.0 (5) vs 4.0 (6)
        assert_eq!(f32_to_fp4_cpu(5.0), 4.0); // 4.0 (6) vs 6.0 (7)
        assert_eq!(f32_to_fp4_cpu(-0.75), -1.0);
        assert_eq!(f32_to_fp4_cpu(6.5), 6.0);
    }

    #[test]
    fn fp4_act_quant_snap_bit_exact() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();

        for (block_size, scale_e4m3) in [(32usize, 0i32), (16, 1)] {
            let n = 8 * block_size;
            let mut x = rand_vec(n, 1234 + block_size as u64);
            x[0] = 900.0; // exercise clamp
            for v in x.iter_mut().take(block_size).skip(1) {
                *v *= 1e-8; // exercise amax floor on group 0? (x[0] keeps amax high)
            }

            let mut expected = x.clone();
            for g in expected.chunks_exact_mut(block_size) {
                let amax_raw = g.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let s = if scale_e4m3 == 1 {
                    let amax = amax_raw.max(6.0 * 2f32.powi(-9));
                    ferroseek_core::quant::e4m3_to_f32(f32_to_e4m3_bits(amax / 6.0))
                } else {
                    let amax = amax_raw.max(6.0 * 2f32.powi(-126));
                    2f32.powf((amax / 6.0).log2().ceil())
                };
                for v in g.iter_mut() {
                    let y = (*v / s).clamp(-6.0, 6.0);
                    *v = f32_to_fp4_cpu(y) * s;
                }
            }

            let mut d_x = DeviceBuffer::from_slice(&x).unwrap();
            let func = cuda
                .fp4_actquant
                .get_function("fp4_act_quant_snap")
                .unwrap();
            let stream = &cuda.stream;
            unsafe {
                launch!(
                    func<<<8u32, 32, 0, stream>>>(
                        d_x.as_device_ptr(),
                        n as i32,
                        block_size as i32,
                        scale_e4m3
                    )
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            d_x.copy_to(&mut x).unwrap();
            assert_eq!(
                x, expected,
                "fp4 snap block={block_size} e4m3={scale_e4m3} not exact"
            );
        }
    }

    /// Pack f32 → (fp4 bytes, e8m0 scales) per 1×32 K-block (low nibble even).
    fn quantize_fp4(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let kb = x.len() / 32;
        let mut q = vec![0u8; x.len() / 2];
        let mut s = vec![0u8; kb];
        for (b, g) in x.chunks_exact(32).enumerate() {
            let amax = g
                .iter()
                .fold(0.0f32, |a, v| a.max(v.abs()))
                .max(6.0 * 2f32.powi(-126));
            let scale = 2f32.powf((amax / 6.0).log2().ceil());
            s[b] = (scale.log2() + 127.0) as u8;
            for k in 0..16 {
                let lo = ferroseek_core::quant::FP4_TABLE
                    .iter()
                    .position(|&t| t == f32_to_fp4_cpu((g[2 * k] / scale).clamp(-6.0, 6.0)))
                    .unwrap() as u8;
                let hi = ferroseek_core::quant::FP4_TABLE
                    .iter()
                    .position(|&t| t == f32_to_fp4_cpu((g[2 * k + 1] / scale).clamp(-6.0, 6.0)))
                    .unwrap() as u8;
                q[b * 16 + k] = lo | (hi << 4);
            }
        }
        (q, s)
    }

    #[test]
    fn fp4_expert_gemm_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (m_dim, n_dim, k_dim) = (2usize, 64usize, 4096usize);
        let kb = k_dim / 32;

        let a_f32 = rand_vec(m_dim * k_dim, 21);
        let b_f32 = rand_vec(n_dim * k_dim, 22);
        let (a_q, a_s) = quantize_fp8(&a_f32);
        let (b_q, b_s) = quantize_fp4(&b_f32);

        // CPU mirror: fp4 table values are exactly e4m3-representable, so
        // the spec's fp4→f32→fp8 cast is lossless and dots run in f32.
        let mut expected = vec![0.0f32; m_dim * n_dim];
        for m in 0..m_dim {
            for n in 0..n_dim {
                let mut acc = 0.0f32;
                for b in 0..kb {
                    let mut dot = 0.0f32;
                    for k in 0..16 {
                        let byte = b_q[n * (k_dim / 2) + b * 16 + k];
                        let av0 =
                            ferroseek_core::quant::e4m3_to_f32(a_q[m * k_dim + b * 32 + 2 * k]);
                        let av1 =
                            ferroseek_core::quant::e4m3_to_f32(a_q[m * k_dim + b * 32 + 2 * k + 1]);
                        dot += av0 * ferroseek_core::quant::FP4_TABLE[(byte & 0x0F) as usize];
                        dot += av1 * ferroseek_core::quant::FP4_TABLE[(byte >> 4) as usize];
                    }
                    let sa = ferroseek_core::quant::e8m0_to_f32(a_s[m * kb + b]);
                    let sb = ferroseek_core::quant::e8m0_to_f32(b_s[n * kb + b]);
                    acc += dot * sa * sb;
                }
                expected[m * n_dim + n] = acc;
            }
        }

        let d_a = DeviceBuffer::from_slice(&a_q).unwrap();
        let d_as = DeviceBuffer::from_slice(&a_s).unwrap();
        let d_b = DeviceBuffer::from_slice(&b_q).unwrap();
        let d_bs = DeviceBuffer::from_slice(&b_s).unwrap();
        let mut d_c = DeviceBuffer::from_slice(&vec![0.0f32; m_dim * n_dim]).unwrap();
        let func = cuda
            .fp4_expert_gemm
            .get_function("fp4_expert_gemm")
            .unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<(n_dim as u32, m_dim as u32), 256, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_as.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_bs.as_device_ptr(),
                    d_c.as_device_ptr(),
                    m_dim as i32,
                    n_dim as i32,
                    k_dim as i32
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; m_dim * n_dim];
        d_c.copy_to(&mut got).unwrap();

        let max_rel: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs() / e.abs().max(1.0))
            .fold(0.0, f32::max);
        assert!(max_rel < 1e-4, "fp4_expert_gemm max rel err {max_rel}");
    }

    #[test]
    fn hc_block_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (n_tokens, hc_dim, mix_hc) = (3usize, 20480usize, 24usize);
        let eps = 1e-6f32;

        let xf = rand_vec(n_tokens * hc_dim, 555);
        let hc_fn = rand_vec(mix_hc * hc_dim, 556);
        let hc_base = rand_vec(mix_hc, 557);
        let hc_scale = rand_vec(3, 558);

        // CPU mirror (spec §2)
        let mut exp_pre = vec![0.0f32; n_tokens * 4];
        let mut exp_post = vec![0.0f32; n_tokens * 4];
        let mut exp_comb = vec![0.0f32; n_tokens * 16];
        for t in 0..n_tokens {
            let x = &xf[t * hc_dim..(t + 1) * hc_dim];
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / hc_dim as f32;
            let rstd = 1.0 / (ms + 1e-20).sqrt();
            let mut mixes = [0.0f32; 24];
            for (row, m) in mixes.iter_mut().enumerate() {
                let w = &hc_fn[row * hc_dim..(row + 1) * hc_dim];
                let dot: f32 = x.iter().zip(w).map(|(a, b)| a * b).sum();
                *m = dot * rstd;
            }
            let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
            for j in 0..4 {
                exp_pre[t * 4 + j] = sig(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
                exp_post[t * 4 + j] = 2.0 * sig(mixes[4 + j] * hc_scale[1] + hc_base[4 + j]);
            }
            let mut comb = [0.0f32; 16];
            for j in 0..4 {
                for k in 0..4 {
                    comb[j * 4 + k] = mixes[8 + j * 4 + k] * hc_scale[2] + hc_base[8 + j * 4 + k];
                }
            }
            for j in 0..4 {
                let mx = (0..4)
                    .map(|k| comb[j * 4 + k])
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = (0..4).map(|k| (comb[j * 4 + k] - mx).exp()).sum();
                for k in 0..4 {
                    comb[j * 4 + k] = (comb[j * 4 + k] - mx).exp() / sum + eps;
                }
            }
            for k in 0..4 {
                let cs: f32 = (0..4).map(|j| comb[j * 4 + k]).sum::<f32>() + eps;
                for j in 0..4 {
                    comb[j * 4 + k] /= cs;
                }
            }
            for _ in 0..19 {
                for j in 0..4 {
                    let rs: f32 = (0..4).map(|k| comb[j * 4 + k]).sum::<f32>() + eps;
                    for k in 0..4 {
                        comb[j * 4 + k] /= rs;
                    }
                }
                for k in 0..4 {
                    let cs: f32 = (0..4).map(|j| comb[j * 4 + k]).sum::<f32>() + eps;
                    for j in 0..4 {
                        comb[j * 4 + k] /= cs;
                    }
                }
            }
            exp_comb[t * 16..(t + 1) * 16].copy_from_slice(&comb);
        }

        let d_xf = DeviceBuffer::from_slice(&xf).unwrap();
        let d_fn = DeviceBuffer::from_slice(&hc_fn).unwrap();
        let d_base = DeviceBuffer::from_slice(&hc_base).unwrap();
        let d_scale = DeviceBuffer::from_slice(&hc_scale).unwrap();
        let mut d_pre = DeviceBuffer::from_slice(&vec![0.0; n_tokens * 4]).unwrap();
        let mut d_post = DeviceBuffer::from_slice(&vec![0.0; n_tokens * 4]).unwrap();
        let mut d_comb = DeviceBuffer::from_slice(&vec![0.0; n_tokens * 16]).unwrap();
        let func = cuda.hc.get_function("hc_block").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<n_tokens as u32, 256, 0, stream>>>(
                    d_xf.as_device_ptr(),
                    d_fn.as_device_ptr(),
                    d_base.as_device_ptr(),
                    d_scale.as_device_ptr(),
                    d_pre.as_device_ptr(),
                    d_post.as_device_ptr(),
                    d_comb.as_device_ptr(),
                    n_tokens as i32,
                    hc_dim as i32,
                    mix_hc as i32,
                    eps
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got_pre = vec![0.0f32; n_tokens * 4];
        let mut got_post = vec![0.0f32; n_tokens * 4];
        let mut got_comb = vec![0.0f32; n_tokens * 16];
        d_pre.copy_to(&mut got_pre).unwrap();
        d_post.copy_to(&mut got_post).unwrap();
        d_comb.copy_to(&mut got_comb).unwrap();

        for (name, got, exp) in [
            ("pre", &got_pre, &exp_pre),
            ("post", &got_post, &exp_post),
            ("comb", &got_comb, &exp_comb),
        ] {
            let max_err: f32 = got
                .iter()
                .zip(exp.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0, f32::max);
            assert!(max_err < 1e-5, "hc {name} max err {max_err}");
        }
    }

    #[test]
    fn sparse_attn_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (n_heads, dim, n_rows) = (4usize, 512usize, 300usize);
        let scale = 512f32.powf(-0.5);

        let q = rand_vec(n_heads * dim, 900);
        let kv = rand_vec(n_rows * dim, 901);
        let sink = rand_vec(n_heads, 902);
        // selection: valid, invalid (-1), repeated, unsorted
        let mut idxs: Vec<i32> = (0..250i32).collect();
        idxs[5] = -1;
        idxs[100] = 7;
        idxs.push(299);
        idxs.push(-1);
        let n_sel = idxs.len();

        // CPU mirror: exact online softmax, same order
        let mut expected = vec![0.0f32; n_heads * dim];
        for h in 0..n_heads {
            let qh = &q[h * dim..(h + 1) * dim];
            let (mut m, mut s) = (-1e30f32, 0.0f32);
            let mut acc = vec![0.0f32; dim];
            for &idx in &idxs {
                let logit = if idx >= 0 {
                    let row = &kv[idx as usize * dim..(idx as usize + 1) * dim];
                    qh.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() * scale
                } else {
                    -1e30
                };
                let mnew = m.max(logit);
                let corr = (m - mnew).exp();
                let w = (logit - mnew).exp();
                s = s * corr + w;
                for d in 0..dim {
                    acc[d] = acc[d] * corr
                        + w * if idx >= 0 {
                            kv[idx as usize * dim + d]
                        } else {
                            0.0
                        };
                }
                m = mnew;
            }
            s += (sink[h] - m).exp();
            for d in 0..dim {
                expected[h * dim + d] = acc[d] / s;
            }
        }

        let d_q = DeviceBuffer::from_slice(&q).unwrap();
        let d_kv = DeviceBuffer::from_slice(&kv).unwrap();
        let d_idx = DeviceBuffer::from_slice(&idxs).unwrap();
        let d_sink = DeviceBuffer::from_slice(&sink).unwrap();
        let mut d_out = DeviceBuffer::from_slice(&vec![0.0; n_heads * dim]).unwrap();
        let func = cuda.attn.get_function("sparse_attn").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<n_heads as u32, 128, 0, stream>>>(
                    d_q.as_device_ptr(),
                    d_kv.as_device_ptr(),
                    d_idx.as_device_ptr(),
                    d_sink.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n_heads as i32,
                    dim as i32,
                    n_sel as i32,
                    scale
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n_heads * dim];
        d_out.copy_to(&mut got).unwrap();
        let max_err: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-4, "sparse_attn max err {max_err}");
    }

    fn f32_to_bf16_bits(v: f32) -> u16 {
        ferroseek_core::quant::f32_to_bf16(v)
    }

    #[test]
    fn moe_route_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (dim, n_experts, top_k) = (256usize, 384usize, 6usize);
        let route_scale = 1.5f32;

        let x = rand_vec(dim, 1001);
        let gate_f32 = rand_vec(n_experts * dim, 1002);
        let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let bias = rand_vec(n_experts, 1003);

        // CPU mirror
        let mut scores: Vec<f32> = (0..n_experts)
            .map(|e| {
                let dot: f32 = (0..dim)
                    .map(|d| x[d] * ferroseek_core::quant::bf16_to_f32(gate_bf16[e * dim + d]))
                    .sum();
                let sp = if dot > 20.0 {
                    dot
                } else {
                    (dot.exp() + 1.0).ln()
                };
                sp.sqrt()
            })
            .collect();
        let mut taken = vec![false; n_experts];
        let mut exp_idx = Vec::new();
        let mut exp_w = Vec::new();
        for _ in 0..top_k {
            let mut best = !0usize;
            let mut best_v = f32::NEG_INFINITY;
            for e in 0..n_experts {
                if taken[e] {
                    continue;
                }
                let v = scores[e] + bias[e];
                if v > best_v {
                    best_v = v;
                    best = e;
                }
            }
            taken[best] = true;
            exp_idx.push(best as i32);
            exp_w.push(scores[best]);
        }
        let wsum: f32 = exp_w.iter().sum();
        for w in exp_w.iter_mut() {
            *w = *w / (wsum + 1e-20) * route_scale;
        }

        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_g = DeviceBuffer::from_slice(&gate_bf16).unwrap();
        let d_b = DeviceBuffer::from_slice(&bias).unwrap();
        let mut d_w = DeviceBuffer::from_slice(&vec![0.0; top_k]).unwrap();
        let mut d_i = DeviceBuffer::from_slice(&vec![0i32; top_k]).unwrap();
        let func = cuda.moe.get_function("moe_route").unwrap();
        let stream = &cuda.stream;
        let shared = (n_experts * 4) as u32;
        unsafe {
            launch!(
                func<<<1, 384, shared, stream>>>(
                    d_x.as_device_ptr(),
                    d_g.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_w.as_device_ptr(),
                    d_i.as_device_ptr(),
                    dim as i32,
                    n_experts as i32,
                    top_k as i32,
                    route_scale
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got_w = vec![0.0f32; top_k];
        let mut got_i = vec![0i32; top_k];
        d_w.copy_to(&mut got_w).unwrap();
        d_i.copy_to(&mut got_i).unwrap();

        assert_eq!(got_i, exp_idx, "route indices differ");
        let max_err: f32 = got_w
            .iter()
            .zip(&exp_w)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-5, "route weights max err {max_err}");
    }

    #[test]
    fn swiglu_limit10_matches_cpu() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let n = 4096usize;
        let gate = rand_vec(n, 2001);
        let up = rand_vec(n, 2002);
        let rw = 0.37f32;
        let mut expected = vec![0.0f32; n];
        for i in 0..n {
            let g = gate[i].min(10.0);
            let u = up[i].clamp(-10.0, 10.0);
            let silu = g * (1.0 / (1.0 + (-g).exp()));
            expected[i] = silu * u * rw;
        }
        let d_g = DeviceBuffer::from_slice(&gate).unwrap();
        let d_u = DeviceBuffer::from_slice(&up).unwrap();
        let mut d_o = DeviceBuffer::from_slice(&vec![0.0; n]).unwrap();
        let func = cuda.moe.get_function("swiglu_limit10").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<(n as u32).div_ceil(256), 256, 0, stream>>>(
                    d_g.as_device_ptr(),
                    d_u.as_device_ptr(),
                    d_o.as_device_ptr(),
                    n as i32,
                    rw
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n];
        d_o.copy_to(&mut got).unwrap();
        let max_err: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-6, "swiglu max err {max_err}");
    }

    #[test]
    fn compressor_finish_and_bf16_snap_match_cpu() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let dim = 512usize;

        let kv = rand_vec(2 * dim, 3001);
        let score = rand_vec(2 * dim, 3002);
        let mut expected = vec![0.0f32; dim];
        for d in 0..dim {
            let (s0, s1) = (score[d], score[dim + d]);
            let mx = s0.max(s1);
            let (e0, e1) = ((s0 - mx).exp(), (s1 - mx).exp());
            expected[d] = (kv[d] * e0 + kv[dim + d] * e1) / (e0 + e1);
        }
        // bf16 snap
        for v in expected.iter_mut() {
            let bits = v.to_bits();
            let snapped = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
            *v = f32::from_bits(snapped);
        }

        let d_kv = DeviceBuffer::from_slice(&kv).unwrap();
        let d_score = DeviceBuffer::from_slice(&score).unwrap();
        let mut d_lat = DeviceBuffer::from_slice(&vec![0.0; dim]).unwrap();
        let finish = cuda.compressor.get_function("compressor_finish").unwrap();
        let snap = cuda.compressor.get_function("bf16_snap").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                finish<<<dim.div_ceil(256) as u32, 256, 0, stream>>>(
                    d_kv.as_device_ptr(),
                    d_score.as_device_ptr(),
                    d_lat.as_device_ptr(),
                    dim as i32
                )
            )
            .unwrap();
            launch!(snap<<<dim.div_ceil(256) as u32, 256, 0, stream>>>(
                d_lat.as_device_ptr(),
                dim as i32
            ))
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; dim];
        d_lat.copy_to(&mut got).unwrap();
        assert_eq!(got, expected, "compressor finish/snap not exact");
    }

    #[test]
    fn indexer_score_matches_cpu_mirror() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (n_heads, head_dim, n_rows) = (32usize, 128usize, 96usize);

        let q = rand_vec(n_heads * head_dim, 4001);
        let index_k = rand_vec(n_rows * head_dim, 4002);
        let weights = rand_vec(n_heads, 4003);

        let mut expected = vec![0.0f32; n_rows];
        for t in 0..n_rows {
            let k = &index_k[t * head_dim..(t + 1) * head_dim];
            for h in 0..n_heads {
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let dot: f32 = qh.iter().zip(k).map(|(a, b)| a * b).sum();
                expected[t] += dot.max(0.0) * weights[h];
            }
        }

        let d_q = DeviceBuffer::from_slice(&q).unwrap();
        let d_k = DeviceBuffer::from_slice(&index_k).unwrap();
        let d_w = DeviceBuffer::from_slice(&weights).unwrap();
        let mut d_o = DeviceBuffer::from_slice(&vec![0.0; n_rows]).unwrap();
        let func = cuda.indexer.get_function("indexer_score").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<n_rows as u32, 32, 0, stream>>>(
                    d_q.as_device_ptr(),
                    d_k.as_device_ptr(),
                    d_w.as_device_ptr(),
                    d_o.as_device_ptr(),
                    n_heads as i32,
                    head_dim as i32,
                    n_rows as i32
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n_rows];
        d_o.copy_to(&mut got).unwrap();
        let max_err: f32 = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-4, "indexer_score max err {max_err}");
    }

    #[test]
    fn f32_and_bf16_gemv_match_cpu() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (rows, cols) = (64usize, 2048usize);

        let w = rand_vec(rows * cols, 5001);
        let wb: Vec<u16> = w.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let x = rand_vec(cols, 5002);

        let mut expected = vec![0.0f32; rows];
        let mut expected_bf = vec![0.0f32; rows];
        for r in 0..rows {
            expected[r] = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            expected_bf[r] = (0..cols)
                .map(|c| ferroseek_core::quant::bf16_to_f32(wb[r * cols + c]) * x[c])
                .sum();
        }

        let d_w = DeviceBuffer::from_slice(&w).unwrap();
        let d_wb = DeviceBuffer::from_slice(&wb).unwrap();
        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let mut d_y = DeviceBuffer::from_slice(&vec![0.0; rows]).unwrap();
        let f32g = cuda.gemv.get_function("f32_gemv").unwrap();
        let bfg = cuda.gemv.get_function("bf16_gemv").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(f32g<<<rows as u32, 256, 0, stream>>>(
                d_w.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                rows as i32,
                cols as i32
            ))
            .unwrap();
            stream.synchronize().unwrap();
            let mut got = vec![0.0f32; rows];
            d_y.copy_to(&mut got).unwrap();
            let e: f32 = got
                .iter()
                .zip(&expected)
                .map(|(g, e)| (g - e).abs())
                .fold(0.0, f32::max);
            assert!(e < 1e-4, "f32_gemv max err {e}");

            launch!(bfg<<<rows as u32, 256, 0, stream>>>(
                d_wb.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                rows as i32,
                cols as i32
            ))
            .unwrap();
            stream.synchronize().unwrap();
            d_y.copy_to(&mut got).unwrap();
            let e: f32 = got
                .iter()
                .zip(&expected_bf)
                .map(|(g, e)| (g - e).abs())
                .fold(0.0, f32::max);
            assert!(e < 1e-4, "bf16_gemv max err {e}");
        }
    }

    #[test]
    fn engram_gather_and_merge_match_cpu() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let (table_rows, cols, n_sel, dim) = (512usize, 256usize, 24usize, 512usize);

        // fp8 table: random f32 -> quantize per row
        let table_f32 = rand_vec(table_rows * cols, 6001);
        let (table_q, table_s) = quantize_fp8(&table_f32);
        let idxs: Vec<i64> = (0..n_sel as i64)
            .map(|i| (i * 37 + 11) % table_rows as i64)
            .collect();

        // CPU gather mirror
        let mut exp_rows = vec![0.0f32; n_sel * cols];
        for (r, &idx) in idxs.iter().enumerate() {
            for c in 0..cols {
                exp_rows[r * cols + c] =
                    ferroseek_core::quant::e4m3_to_f32(table_q[idx as usize * cols + c])
                        * ferroseek_core::quant::e8m0_to_f32(
                            table_s[idx as usize * (cols / 32) + c / 32],
                        );
            }
        }

        let d_t = DeviceBuffer::from_slice(&table_q).unwrap();
        let d_s = DeviceBuffer::from_slice(&table_s).unwrap();
        let d_i = DeviceBuffer::from_slice(&idxs).unwrap();
        let mut d_o = DeviceBuffer::from_slice(&vec![0.0; n_sel * cols]).unwrap();
        let gather = cuda.engram.get_function("gather_rows_fp8").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                gather<<<n_sel as u32, 256, 0, stream>>>(
                    d_t.as_device_ptr(),
                    d_s.as_device_ptr(),
                    d_i.as_device_ptr(),
                    d_o.as_device_ptr(),
                    n_sel as i32,
                    cols as i32
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got = vec![0.0f32; n_sel * cols];
        d_o.copy_to(&mut got).unwrap();
        assert_eq!(got, exp_rows, "gather not exact");

        // merge: 4 hc copies
        let mut h = rand_vec(4 * dim, 6002);
        let key = rand_vec(4 * dim, 6003);
        let value = rand_vec(dim, 6004);
        let w = rand_vec(dim, 6005);
        let mut expected = h.clone();
        for j in 0..4 {
            let hj = &h[j * dim..(j + 1) * dim];
            let kj = &key[j * dim..(j + 1) * dim];
            let mut dot = 0.0f32;
            let mut hsq = 0.0f32;
            let mut ksq = 0.0f32;
            for d in 0..dim {
                dot += hj[d] * w[d] * kj[d];
                hsq += hj[d] * hj[d];
                ksq += kj[d] * kj[d];
            }
            let rstd_h = 1.0 / (hsq / dim as f32 + 1e-20).sqrt();
            let rstd_k = 1.0 / (ksq / dim as f32 + 1e-20).sqrt();
            let d01 = dot * rstd_h * rstd_k * (dim as f32).powf(-0.5);
            let gate = 1.0 / (1.0 + (-d01.abs().max(1e-6).sqrt().copysign(d01)).exp());
            for d in 0..dim {
                expected[j * dim + d] += gate * value[d];
            }
        }

        let mut d_h = DeviceBuffer::from_slice(&h).unwrap();
        let d_k = DeviceBuffer::from_slice(&key).unwrap();
        let d_v = DeviceBuffer::from_slice(&value).unwrap();
        let d_w = DeviceBuffer::from_slice(&w).unwrap();
        let merge = cuda.engram.get_function("engram_merge").unwrap();
        let inv = (dim as f32).powf(-0.5);
        unsafe {
            launch!(
                merge<<<4u32, 256, 0, stream>>>(
                    d_h.as_device_ptr(),
                    d_k.as_device_ptr(),
                    d_v.as_device_ptr(),
                    d_w.as_device_ptr(),
                    dim as i32,
                    inv
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        d_h.copy_to(&mut h).unwrap();
        let max_err: f32 = h
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-5, "engram merge max err {max_err}");
    }

    #[test]
    fn argmax_matches_cpu_incl_ties() {
        let guard = Cuda::global();
        let cuda = guard.as_ref().unwrap();
        let n = 129280usize;
        let mut x = rand_vec(n, 7001);
        x[500] = 0.99;
        x[90000] = 0.99; // tie: lowest index wins
        x[129279] = -100.0;

        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let mut d_i = DeviceBuffer::from_slice(&[0i32; 1]).unwrap();
        let mut d_v = DeviceBuffer::from_slice(&[0.0f32; 1]).unwrap();
        let func = cuda.sample.get_function("argmax_rows").unwrap();
        let stream = &cuda.stream;
        unsafe {
            launch!(
                func<<<1, 256, 0, stream>>>(
                    d_x.as_device_ptr(),
                    d_i.as_device_ptr(),
                    d_v.as_device_ptr(),
                    n as i32
                )
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut got_i = [0i32; 1];
        d_i.copy_to(&mut got_i).unwrap();
        assert_eq!(got_i[0], 500, "argmax index (tie -> lowest)");
    }
}
