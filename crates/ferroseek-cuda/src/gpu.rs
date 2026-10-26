//! Driver context, device probe and kernel module loading.

use cust::context::CurrentContext;
use cust::prelude::*;
use std::sync::Mutex;

const DEQUANT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/dequant.ptx"));

pub struct Cuda {
    pub context: Context,
    pub device: Device,
    pub stream: Stream,
    pub dequant: Module,
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
            *guard = Some(Cuda {
                context,
                device,
                stream,
                dequant,
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
}
