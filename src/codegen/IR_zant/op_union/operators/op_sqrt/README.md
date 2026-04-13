# Sqrt

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Sqrt.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_sqrt.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_sqrt.zig` — unit tests for the sqrt operator.
- `utils_sqrt.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_sqrt.zig` — runtime math implementation (lean and standard variants) called by generated code.
