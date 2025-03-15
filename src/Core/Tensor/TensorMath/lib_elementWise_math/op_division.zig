const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

/// Performs Element-wise binary division of two tensors.
pub fn div(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T)) !Tensor(T) {
    if (lhs.size != rhs.size) {
        return TensorError.MismatchedShape;
    }

    const allocator = lhs.allocator;
    var result = try Tensor(T).fromShape(allocator, lhs.shape);

    div_lean(T, lhs, rhs, &result);

    return result;
}
// --------- lean DIV
pub inline fn div_lean(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T), result: *Tensor(T)) void {
    for (0..lhs.size) |i| {
        result.data[i] = lhs.data[i] / rhs.data[i];
    }
}
