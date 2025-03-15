const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type

/// Compute element-wise the hyperbolic tangent of the given tensor.
pub fn tanh(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    // Verify that T is among the supported types:
    // tensor(double), tensor(float), tensor(float16)
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in tanh_lean");
    };

    // Allocating output tensor with the same shape of the input
    var result = try Tensor(T).fromShape(input.allocator, input.shape);

    tanh_lean(T, input, &result);

    return result;
}

// --------- lean TANH
pub inline fn tanh_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) void {
    // Compute tanh(x) for each element of the tensor
    for (0..input.size) |i| {
        result.data[i] = std.math.tanh(input.data[i]);
    }
}
