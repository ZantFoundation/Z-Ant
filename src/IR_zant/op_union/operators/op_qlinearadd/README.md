# QLinearAdd

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QLinearAdd.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_qlinearadd.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `utils_qlinearadd.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_qlinearadd.zig` — runtime math implementation (lean and standard variants) called by generated code.
