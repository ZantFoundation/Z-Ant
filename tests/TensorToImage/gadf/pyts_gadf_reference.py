"""
Generates GADF test vectors using pyts for cross-validation with the Zig implementation.

Run this script *before* `zig build test` to produce gadf_test_vectors.json.
The Zig test "GADF pyts cross-validation" reads that file and checks its own
output matches pyts cell-by-cell.

Requirements:
    pip install pyts numpy

Usage (from project root):
    python tests/TensorToImage/pyts_gadf_reference.py
"""

import json
import sys


def main():
    try:
        import numpy as np
        from pyts.image import GramianAngularField
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pyts numpy")
        sys.exit(1)

    # GramianAngularField with method='difference' and sample_range=(-1, 1)
    # matches our Zig gadf() with NormRange.MinusOneToOne.
    gaf = GramianAngularField(method="difference", sample_range=(-1, 1))

    rng = np.random.default_rng(seed=42)
    test_vectors = []

    # --- Fixed series (reproducible, hand-verifiable) ---
    fixed = [
        ("fixed_4",  [0.1,  0.5, -0.3,  0.8]),
        ("fixed_5",  [1.0,  2.0,  3.0,  4.0,  5.0]),
        ("fixed_6",  [-2.0, 0.0,  1.0,  0.5, -1.0,  2.0]),
        ("fixed_8",  [0.3, -0.7,  0.1,  0.9, -0.2,  0.5, -0.8,  0.4]),
    ]
    for label, series in fixed:
        x = np.array(series, dtype=np.float64).reshape(1, -1)
        G = gaf.fit_transform(x)[0]
        test_vectors.append({
            "label": label,
            "input": series,
            "expected": G.flatten().tolist(),
        })

    # --- Random series with fixed seed (varied lengths) ---
    for n in [5, 8, 10, 15, 20]:
        series = rng.uniform(-3.0, 3.0, size=n).tolist()
        x = np.array(series, dtype=np.float64).reshape(1, -1)
        G = gaf.fit_transform(x)[0]
        test_vectors.append({
            "label": f"random_len{n}",
            "input": series,
            "expected": G.flatten().tolist(),
        })

    out_path = "tests/TensorToImage/gadf_test_vectors.json"
    with open(out_path, "w") as f:
        json.dump(test_vectors, f, indent=2)

    print(f"Wrote {len(test_vectors)} test vectors -> {out_path}")
    for v in test_vectors:
        n = len(v["input"])
        print(f"  [{v['label']}]  n={n}  matrix={n}x{n}  cells={n*n}")


if __name__ == "__main__":
    main()
