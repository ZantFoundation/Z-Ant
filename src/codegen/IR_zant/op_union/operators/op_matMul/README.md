# MatMul

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__MatMul.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_matMul.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_matMul.zig` — unit tests for the matMul operator.
- `utils_matMul.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_blocked_mat_mul.zig` — runtime math implementation variant (`blocked_mat_mul`).
- `zant_matMul.zig` — runtime math implementation (lean and standard variants) called by generated code.
