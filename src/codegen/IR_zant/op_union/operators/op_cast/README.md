# Cast

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__Cast.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### Integer → integer conversions must wrap, not clamp

ONNX `Cast` between integer types follows numpy `astype` / C-style semantics:
narrowing truncates (modular wrap), widening sign-extends signed sources. The
ONNX reference implementation is built on `numpy.astype`, so any deviation
fails the per-op tests in `tests/CodeGen/Python-ONNX/`.

Symptom of clamping instead of wrapping: an `int32` input of `-91` produces
`uint8` `0` instead of the expected `165` (= `256 - 91`). The conversion logic
lives in `castIntToInt` in `zant_cast.zig` — keep using bitcast/truncate, do
not introduce `std.math.clamp` for integer→integer paths.

### Float → integer

For float→int the ONNX spec leaves overflow behavior implementation-defined.
We currently saturate (clamp) for `f32 → i8` / `f32 → u8`. Revisit only if a
real model relies on wrap semantics there.

## Files

- `op_cast.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `zant_cast.zig` — runtime math implementation (lean and standard variants) called by generated code.
