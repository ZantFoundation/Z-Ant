# TopK

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__TopK.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### Use OUTPUT strides when writing values/indices

`topk_lean` reads from `input` (shape `[..., axis_dim, ...]`) and writes to
`values_output` / `indices_output` (shape `[..., k, ...]`). Both write loops
must use *output* strides — using the input strides causes the per-row write
position to skip past the smaller output buffer (e.g. for input `[3,5]`,
output `[3,2]`, the second outer index lands at linear offset 5/6 in a
6-element buffer). The fix is to compute a separate `out_strides` array with
the axis dim replaced by `k` and use it when writing.

### `compute_output_shape` must read `K` from the initializer

`K` is an INT64 input, not an attribute. Defaulting to `k=1` for shape
inference produces an undersized output buffer at codegen time and the
actual `topk_lean` call then writes out of bounds. Read the value from
`self.input_K.ptr.?.get_data_as(i64)[0]` when available.

### TopK has two outputs — graph-level multi-output is not supported

The current `predict.zig` writer only supports a single graph output, so
the test models in `tests/CodeGen/Python-ONNX/operators/topK.py` expose
only `values` as the graph output and leave `indices` dangling on the node.
Codegen still emits both tensors; the `indices` LINK tensor is freed
immediately after the node by the execution plan (see
`shouldDeallocateAfter` in `src/codegen/cg_v1/predict/plan.zig` — dangling
LINK outputs whose `last_use_step` is `null` are scheduled for free at
their definition step).

## Files

- `op_topk.zig` — IR operator struct: parses the ONNX node, performs shape inference, and emits the codegen call via `write_op`.
- `utils_topk.zig` — shape-inference and attribute-parsing helpers used by the IR layer.
- `zant_topk.zig` — runtime math implementation (lean and standard variants) called by generated code.
