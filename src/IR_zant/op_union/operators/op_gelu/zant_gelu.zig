const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

pub fn get_gelu_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}

pub fn gelu(comptime T: anytype, input: *Tensor(T), approximate: ?[]const u8) !Tensor(T) {
    //check type
    comptime if (!(std.meta.eql(T, f16) or std.meta.eql(T, f32) or std.meta.eql(T, f64))) {
        @compileError("unsupported type in Gelu");
    };

    //check approximate
    if (!(std.mem.eql(u8, approximate.?, "tanh") or std.mem.eql(u8, approximate.?, "none"))) {
        return error.ApproximateError;
    }

    //compute outputshape
    const output_shape = try get_gelu_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    //call lean version
    try gelu_lean(T, input, approximate, &output);

    return output;
}

pub fn gelu_lean(comptime T: type, input: *Tensor(T), approximate: ?[]const u8, output: *Tensor(T)) !void {
    if (input.data.len != output.data.len) {
        return error.ShapeMismatch;
    }

    const sqrt_2 = @sqrt(@as(f32, 2.0));
    const sqrt_2_over_pi = @sqrt(@as(f32, 2.0 / std.math.pi));
    const coeff = @as(f32, 0.044715);

    for (input.data, output.data) |x, *out_val| {
        const x_f32 = @as(f32, @floatCast(x));
        if (std.mem.eql(u8, approximate.?, "tanh")) {
            // x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const x_cubed = x_f32 * x_f32 * x_f32;
            const tanh_arg = sqrt_2_over_pi * (x_f32 + coeff * x_cubed);
            const tanh_val = std.math.tanh(tanh_arg);
            out_val.* = @as(T, @floatCast(x_f32 * 0.5 * (1.0 + tanh_val)));
        } else {
            // x * 0.5 * (1 + erf(x / sqrt(2)))
            const erf_arg = x_f32 / sqrt_2;
            const erf_val = erf(erf_arg);
            out_val.* = @as(T, @floatCast(x_f32 * 0.5 * (1.0 + erf_val)));
        }
    }
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
