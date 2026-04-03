const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
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
    try ceil_lean(T, input, &result);

    return result;
}

// --------- lean CEIL
pub inline fn ceil_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) !void {
    for (0..input.size) |i| {
        const x = input.data[i];
        if (std.math.isNan(x) or std.math.isInf(x)) {
            result.data[i] = x;
        } else {
            result.data[i] = @ceil(x);
        }
    }
}

pub fn get_ceil_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
