# Pad

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Pad.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_pad.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `op_padding.zig` — IR helper module for the pad operator family.
- `op_pads.zig` — IR helper module for the pad operator family.
- `test_pad.zig` — unit tests for the pad operator.
- `utils_pad.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_pad.zig` — runtime math implementation (lean and standard variants) called by generated code.
