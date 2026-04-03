const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

pub fn get_sqrt_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}

pub fn sqrt_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    for (input.data, output.data) |in_val, *out_val| {
        if (in_val < 0) {
            out_val.* = std.math.nan(T);
        } else {
            out_val.* = std.math.pow(T, in_val, 0.5);
        }
    }
}

pub fn sqrt(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in sqrt_lean");
    };

    const output_shape = try get_sqrt_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try sqrt_lean(T, input, &output);
    return output;
}
