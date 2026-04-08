# QLinearMatMul

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QLinearMatMul.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_qlinearmatmul.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_qlinearmatmul.zig` — unit tests for the qlinearmatmul operator.
- `utils_qlinearmatmul.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_qlinearmatmul.zig` — runtime math implementation (lean and standard variants) called by generated code.
