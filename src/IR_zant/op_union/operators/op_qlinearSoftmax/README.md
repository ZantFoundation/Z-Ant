# QLinearSoftmax

This is a Microsoft contrib operator. For the full specification, refer to the official ONNX documentation:

https://onnx.ai/onnx/operators/onnx_microsoft__QLinearSoftmax.html

Please look up the standard for input/output definitions, type constraints, and attribute details before modifying this implementation.

## Troubleshooting

### Do not "adjust" the axis to NCHW

An earlier version of `op_qlinearSoftmax.zig` assumed every input was 4D
`[N, C, H, W]` and rewrote the user-supplied axis as `axis - (4 - rank)`.
For a 2D `[1, 5]` model with `axis=1` this collapsed to axis 0, and softmax
was computed over the singleton batch dim — every output element became
`y_zero_point + 1/y_scale` (clamped). Just normalize negative axes by
adding `rank`, matching the ONNX `Softmax` spec.
