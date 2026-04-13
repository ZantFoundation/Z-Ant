# Reshape

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Reshape.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_reshape.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_reshape.zig` — unit tests for the reshape operator.
- `utils_reshape.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_reshape.zig` — runtime math implementation (lean and standard variants) called by generated code.
- `zant_reshape_f32.zig` — runtime math implementation variant (`reshape_f32`).
