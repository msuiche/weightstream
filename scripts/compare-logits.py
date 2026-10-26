#!/usr/bin/env python3
"""Compare two --logits-raw dirs position by position (GPU vs CPU engine).

Usage: python3 scripts/compare-logits.py <cpu-raw-dir> <gpu-raw-dir>
"""
import os
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
pos = 0
worst = []
while True:
    pa = os.path.join(a_dir, f"pos{pos}.f32")
    pb = os.path.join(b_dir, f"pos{pos}.f32")
    if not os.path.exists(pa) or not os.path.exists(pb):
        break
    a = np.fromfile(pa, dtype="<f4")
    b = np.fromfile(pb, dtype="<f4")
    d = np.abs(a - b)
    worst.append((pos, float(d.max()), int(np.argmax(d)), float(a.max()), float(b.max()),
                  int(np.argmax(a)), int(np.argmax(b))))
    pos += 1
print(f"{'pos':>4} {'max|d|':>12} {'argmax a':>8} {'argmax b':>8} {'agree':>5}")
for p, m, _, am, bm, ta, tb in worst:
    print(f"{p:>4} {m:12.6f} {ta:>8} {tb:>8} {str(ta == tb):>5}")
if worst:
    print(f"max over positions: {max(w[1] for w in worst):.6f} at pos {max(worst, key=lambda w: w[1])[0]}")
