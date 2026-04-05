const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

/// Computes element-wise minimum between two tensors, allocating a new output tensor.
pub fn min_two(comptime T: anytype, a: *Tensor(T), b: *Tensor(T)) !Tensor(T) {
    if (a.size != b.size) return error.InvalidDimensions;

    var output = try Tensor(T).fromShape(a.allocator, a.shape);
    errdefer output.deinit();

    try min_two_lean(T, a, b, &output);
    return output;
}

/// Lean version for two tensors that computes min in-place using a pre-allocated output tensor.
pub fn min_two_lean(comptime T: anytype, a: *Tensor(T), b: *Tensor(T), output: *Tensor(T)) !void {
    if (a.size != b.size or a.size != output.size) return error.InvalidDimensions;

    for (0..output.size) |i| {
        output.data[i] = if (a.data[i] < b.data[i]) a.data[i] else b.data[i];
    }
}
