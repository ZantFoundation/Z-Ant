# useless

This is an internal/custom operator without a direct ONNX standard specification.

For general ONNX operator reference, see: https://onnx.ai/onnx/operators/

## Files

- `op_useless.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
