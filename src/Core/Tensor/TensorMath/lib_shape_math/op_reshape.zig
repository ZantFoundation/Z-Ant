const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// The lean version of this method follows the onnx standard.
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []usize, allowZero: ?bool) !Tensor(T) {
    //TODO: threat allowZero properly
    var total_size: usize = 1;
    for (newShape) |dim| {
        total_size *= dim;
    }
    if (total_size != input.size) {
        return TensorError.InputArrayWrongSize;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, newShape);

    try reshape_lean(T, input, newShape, allowZero, &output);

    return output;
}

/// lean version of the above reshape
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []usize, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero; //TODO: threat allowZero properly

    @memcpy(output.data, input.data);

    for (newShape, 0..) |dim, i| {
        output.shape[i] = dim;
    }
}
