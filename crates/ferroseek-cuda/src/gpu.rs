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
            let fp4_actquant = Module::from_ptx(FP4_ACTQUANT_PTX, &[]).expect("fp4_actquant module");
            let fp4_expert_gemm =
                Module::from_ptx(FP4_EXPERT_GEMM_PTX, &[]).expect("fp4_expert_gemm module");
            let hc = Module::from_ptx(HC_PTX, &[]).expect("hc module");
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
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
            if e >= 15 {
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
        if e >= 15 {
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
            assert_eq!(x, expected, "fp4 snap block={block_size} e4m3={scale_e4m3} not exact");
        }
    }

    /// Pack f32 → (fp4 bytes, e8m0 scales) per 1×32 K-block (low nibble even).
    fn quantize_fp4(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let kb = x.len() / 32;
        let mut q = vec![0u8; x.len() / 2];
        let mut s = vec![0u8; kb];
        for (b, g) in x.chunks_exact(32).enumerate() {
            let amax = g.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(6.0 * 2f32.powi(-126));
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
                        let av0 = ferroseek_core::quant::e4m3_to_f32(a_q[m * k_dim + b * 32 + 2 * k]);
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
        let func = cuda.fp4_expert_gemm.get_function("fp4_expert_gemm").unwrap();
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
                let mx = (0..4).map(|k| comb[j * 4 + k]).fold(f32::NEG_INFINITY, f32::max);
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
}
