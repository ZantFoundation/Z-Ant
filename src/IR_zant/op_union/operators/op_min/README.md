# Min

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Min.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_min.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_min.zig` — unit tests for the min operator.
- `utils_min.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_min.zig` — runtime math implementation (lean and standard variants) called by generated code.
- `zant_min_two.zig` — runtime math implementation variant (`min_two`).
- `zant_reduce_min.zig` — runtime math implementation variant (`reduce_min`).
