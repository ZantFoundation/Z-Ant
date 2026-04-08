# Neg

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Neg.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Files

- `op_neg.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `test_neg.zig` — unit tests for the neg operator.
- `utils_neg.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_neg.zig` — runtime math implementation (lean and standard variants) called by generated code.
- `zant_neg_flip.zig` — runtime math implementation variant (`neg_flip`).
