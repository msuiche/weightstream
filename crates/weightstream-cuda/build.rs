//! Build the .cu kernels to PTX when nvcc is available; otherwise emit an
//! empty manifest so host-only `cargo test` still works (GPU tests skip at
//! runtime).

use std::path::{Path, PathBuf};
use std::process::Command;

fn nvcc() -> Option<PathBuf> {
    for cand in [
        std::env::var("CUDA_HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("bin/nvcc")),
        Some(PathBuf::from("/usr/local/cuda/bin/nvcc")),
    ]
    .into_iter()
    .flatten()
    {
        if cand.exists() {
            return Some(cand);
        }
    }
    Command::new("which")
        .arg("nvcc")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernels_dir = Path::new("kernels");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    // Link search paths for libcuda (driver + toolkit stubs)
    for p in [
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib64/stubs",
    ] {
        println!("cargo:rustc-link-search=native={p}");
    }

    let mut built = Vec::new();
    if let Some(nvcc) = nvcc() {
        let mut entries: Vec<_> = std::fs::read_dir(kernels_dir)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        entries.extend(
            std::fs::read_dir(kernels_dir.join("tilelang"))
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect::<Vec<_>>())
                .unwrap_or_default(),
        );
        for path in entries {
            if path.extension().and_then(|e| e.to_str()) != Some("cu") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            let ptx = out_dir.join(format!("{stem}.ptx"));
            // sm_121 = GB10 (DGX Spark). PTX lets the driver JIT forward.
            // moe.cu compiles with FMA contraction OFF: its gate dots must
            // bit-match the CPU reference's unfused sequential adds — a
            // 1-ULP contraction difference at a near-tie flipped the
            // expert ranking and amplified to a full logit divergence
            // (the synthetic gate's step-1/step-2 flip, root-caused by
            // per-layer capture + dump diffing).
            let mut cmd = Command::new(&nvcc);
            cmd.args(["-ptx", "-O3"]);
            if stem == "moe" {
                cmd.arg("-fmad=false");
            }
            if path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) == Some("tilelang") {
                cmd.arg("-arch=compute_121a").arg("-I").arg("vendor");
            } else {
                cmd.arg("-arch=compute_121");
            }
            cmd.arg("-o").arg(&ptx).arg(&path);
            let status = cmd.status().expect("nvcc run");
            assert!(status.success(), "nvcc failed for {}", path.display());
            built.push(stem);
        }
    } else {
        println!("cargo:warning=nvcc not found; GPU kernels skipped (host-only build)");
    }
    std::fs::write(out_dir.join("kernels.txt"), built.join("\n")).unwrap();
}
