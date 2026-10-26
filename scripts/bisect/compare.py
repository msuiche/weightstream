"""Compare the ferroseek-ref (rust) and model.py (py) per-layer dumps.

Pairs every `*.f32` / `*.i32` dump with its `py_` counterpart (raw
little-endian) and reports max-abs and relative-L2 differences per file.
Rust decodes f32; the reference runs bf16 storage with identical quantized
weights, so a healthy pair agrees to bf16-transport noise (~1e-2 relative);
a semantic break shows up as O(1) at the first mechanism that differs.
`.i32` files (attention indices, expert selection) must match exactly.

Usage: python3 compare.py /bisect/out
"""

import os
import re
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/bisect/out"
# With a second dir: compare py_* dumps of dir1 against py_* of dir2
# (shadow experiment — pure transport-noise amplification, no engine gap).
if len(sys.argv) > 2:
    import glob

    d1, d2 = out, sys.argv[2]
    files = sorted(
        f for f in os.listdir(d1)
        if f.startswith("py_") and (f.endswith(".f32") or f.endswith(".i32"))
    )
    for rf in files:
        pa = os.path.join(d1, rf)
        pb = os.path.join(d2, rf)
        dtype = "<f4" if rf.endswith(".f32") else "<i4"
        a = np.fromfile(pa, dtype=dtype)
        b = np.fromfile(pb, dtype=dtype)
        if a.shape != b.shape:
            print(f"{rf:36} shape {a.shape} vs {b.shape}")
            continue
        if dtype == "<i4":
            nbad = int((a != b).sum())
            if nbad:
                print(f"{rf:36} {nbad}/{a.size} int mismatches")
            continue
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
        print(f"{rf:36} rel-l2 {rel:10.4e}")
    sys.exit(0)

rust_files = sorted(
    f
    for f in os.listdir(out)
    if (f.endswith(".f32") or f.endswith(".i32")) and not f.startswith("py_")
)

rows = []
for rf in rust_files:
    pf = os.path.join(out, "py_" + rf)
    if not os.path.exists(pf):
        rows.append((rf, None, None, None))
        continue
    dtype = "<f4" if rf.endswith(".f32") else "<i4"
    a = np.fromfile(os.path.join(out, rf), dtype=dtype)
    b = np.fromfile(pf, dtype=dtype)
    if a.shape != b.shape:
        rows.append((rf, None, None, f"shape {a.shape} vs {b.shape}"))
        continue
    if dtype == "<i4":
        nbad = int((a != b).sum())
        rows.append((rf, None, None, f"{nbad}/{a.size} mismatch" if nbad else "exact"))
        continue
    d = np.abs(a - b)
    rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
    rows.append((rf, float(d.max()), float(rel), None))


def key(row):
    name = row[0]
    m = re.match(r"(?:d_)?pos(\d+)_(.*?)(\d*)\.(f32|i32)$", name)
    if m:
        return (int(m.group(1)), m.group(2), int(m.group(3) or 0))
    m2 = re.match(r"logits(\d+)\.f32", name)
    if m2:
        return (int(m2.group(1)), "zzlogits", 0)
    return (999, name, 0)


print(f"{'file':36} {'max|d|':>12} {'rel-l2':>10}  note")
worst = 0.0
for row in sorted(rows, key=key):
    name, mx, rel, note = row
    if mx is None:
        print(f"{name:36} {'-':>12} {'-':>10}  {note or 'MISSING py counterpart'}")
        continue
    worst = max(worst, rel)
    print(f"{name:36} {mx:12.4e} {rel:10.4e}")

print(f"\nworst relative L2 across all f32 pairs: {worst:.4e}")