# QLinearAveragePool

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QLinearAveragePool.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_qlinearaveragepool.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_qlinear_pooling.zig` — unit tests (`qlinear_pooling` variant).
- `utils_qlinearaveragepool.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_qlinearaveragepool.zig` — runtime math implementation (lean and standard variants) called by generated code.
