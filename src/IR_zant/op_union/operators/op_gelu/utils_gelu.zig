const std = @import("std");
const zant = @import("zant");
const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_gelu_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}

/// Computes the definite integral of f(x) between a and b using the trapezoidal method
pub fn integrateTrapezoid(
    comptime T: type,
    f: fn (T) T,
    a: T,
    b: T,
    n: usize,
) T {
    const h = (b - a) / @as(T, @floatFromInt(n));
    var sum = (f(a) + f(b)) / @as(T, 2);

    var i: usize = 1;
    while (i < n) : (i += 1) {
        const x = a + h * @as(T, @floatFromInt(i));
        sum += f(x);
    }

    return h * sum;
}

pub fn erfIntegrand(x: f64) f64 {
    return std.math.exp(-x * x);
}

pub fn erf(x: f64) f64 {
    const sqrt_pi_inv = 2.0 / std.math.sqrt(std.math.pi);
    const n = 10000; // Number of subintervals, increase for higher precision
    return sqrt_pi_inv * integrateTrapezoid(f64, erfIntegrand, 0.0, x, n);
}
