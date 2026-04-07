# NonMaxSuppression

For the full specification of this operator, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### Output is `[num_selected, 3]`, not flat

ONNX NMS produces a 2-D tensor whose first dim is the dynamic count of
selected boxes. The Z-Ant codegen pre-allocates a fixed buffer at compile
time, so test models (and any host model) must declare a concrete max
shape `[max_rows, 3]`. A `[None, 3]` value-info collapses to a 1-D `[3]`
in the codegen pipeline and the runtime then writes past the end.

### Bound writes by `data.len / 3`, not `shape[0]`

`non_max_suppression_lean` should cap the number of rows it writes by
`output.data.len / 3`, never trust `output.shape[0]` alone. If the shape
is ever stale or 1-D the data buffer is the only safe truth.
