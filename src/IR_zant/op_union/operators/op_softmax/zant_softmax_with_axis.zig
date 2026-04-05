const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;

const zant_softmax = @import("zant_softmax.zig");

/// Softmax with configurable axis parameter following ONNX specification
pub fn softmax_with_axis(comptime T: anytype, tensor: *Tensor(T), axis: i32) !Tensor(T) {

    //checks
    if (tensor.size <= 0) return error.ZeroSizeTensor;
    if (tensor.shape.len < 2 or tensor.shape.len > 5) return error.InvalidDimensions;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    //compute
    try softmax_with_axis_lean(T, tensor, &output_tensor, axis);
    return output_tensor;
}

pub inline fn softmax_with_axis_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T), axis: i32) !void {
    try zant_softmax.softmax_lean(T, input, output, axis);
}
