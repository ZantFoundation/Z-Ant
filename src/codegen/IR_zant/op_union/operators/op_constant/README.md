# Constant

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Constant.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_constant.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
