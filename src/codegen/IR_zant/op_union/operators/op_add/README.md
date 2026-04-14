# Add

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Add.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_add.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `utils_add.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_add.zig` — runtime math implementation (lean and standard variants) called by generated code.
- `zant_add_list.zig` — runtime math implementation variant (`add_list`).
