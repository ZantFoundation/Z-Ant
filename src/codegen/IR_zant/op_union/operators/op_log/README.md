# Log

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Log.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_log.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_log.zig` — unit tests for the log operator.
- `utils_log.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_log.zig` — runtime math implementation (lean and standard variants) called by generated code.
