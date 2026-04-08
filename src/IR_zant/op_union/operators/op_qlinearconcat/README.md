# QLinearConcat

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QLinearConcat.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### Test models use the standard composite, not the contrib op directly

ORT's contrib `QLinearConcat` shape inference is fragile (it rejects axes
even when they look valid for the concatenated rank), so the fuzz model in
`tests/CodeGen/Python-ONNX/operators/qLinearConcat.py` emits the equivalent
`DequantizeLinear x N → Concat → QuantizeLinear` composite — same approach
already used for `qLinearAdd.py`. This loads cleanly in ORT, ORT computes
the reference output, and Z-Ant's fusion engine can collapse the composite
back to a single `QLinearConcat` if it recognises the pattern. Either way
the runtime values must match the ORT reference.

## Files

- `op_qlinearconcat.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `utils_qlinearconcat.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_qlinearconcat.zig` — runtime math implementation (lean and standard variants) called by generated code.
