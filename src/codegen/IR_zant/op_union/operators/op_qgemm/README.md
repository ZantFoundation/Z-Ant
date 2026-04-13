# QGemm

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QGemm.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### `qgemm_lean` expects B in `[N, K]` layout (`transB=1`)

Z-Ant's `qgemm_lean` (in `op_qlinearmatmul/utils_qlinearmatmul.zig`) is
hard-coded to consume B as `[N, K]`, not the standard `[K, N]`. Test models
in `tests/CodeGen/Python-ONNX/operators/qGemm.py` therefore declare B as
`[N, K]` and emit `transB=1` so ORT computes the matching reference. If you
ever feed a `[K, N]` weight tensor without `transB=1`, the inner matmul
loop will index past the output buffer and SIGABRT.

### TODO: bias `C` is currently ignored

The `op_qgemm.zig` write_op carries `input_C` through but `qgemm_lean` does
not consume it (see comment "bias ignored for now"). The fuzz test sets the
bias to zero so the matmul path is exercised correctly. To restore real
bias support, extend `qgemm_lean` to add the int32 bias to `sum_int` before
the scale/round step (per the ORT contrib spec, C is pre-quantized to
`int32(C / (a_scale * b_scale))`).

## Files

- `op_qgemm.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
