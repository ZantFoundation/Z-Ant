# Transpose

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Transpose.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_transpose.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_transpose.zig` — unit tests for the transpose operator.
- `utils_transpose.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_transpose.zig` — runtime math implementation (lean and standard variants) called by generated code.
- `zant_transpose_variants.zig` — runtime math implementation variant (`transpose_variants`).
