const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type

// --------- standard CEIL
/// Compute element-wise the ceil of the given tensor.
/// If x is integral, +0, -0, NaN, or infinite, x itself is returned.
pub fn ceil(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    // Verify that T is among the supported types:
    // tensor(double), tensor(float), tensor(float16)
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in ceil_lean");
    };

    // Allocate output tensor with the same shape as the input
    var result = try Tensor(T).fromShape(input.allocator, input.shape);

    // Perform element-wise ceil computation
    ceil_lean(T, input, &result);

    return result;
}

// --------- lean CEIL
pub inline fn ceil_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) void {
    for (0..input.size) |i| {
        const x = input.data[i];
        if (std.math.isNan(x) or std.math.isInf(x)) {
            result.data[i] = x;
        } else {
            result.data[i] = @ceil(x);
        }
    }
}
