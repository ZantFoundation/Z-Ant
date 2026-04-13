const std = @import("std");
const IR_zant = @import("IR_zant");
const pkg_allocator = IR_zant.pkg_allocator.allocator;

const Tensor = IR_zant.core.tensor.Tensor;

pub const utils = @import("utils_gelu.zig");

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
    const output_shape = try utils.get_gelu_output_shape(input.shape);
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
            const erf_val = utils.erf(erf_arg);
            out_val.* = @as(T, @floatCast(x_f32 * 0.5 * (1.0 + erf_val)));
        }
    }
}
