const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor; // Import Tensor type

pub fn get_floor_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}

pub fn floor(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16))) {
        @compileError("Unsupported type in floor_lean");
    };

    const output_shape = try get_floor_output_shape(input.shape);
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    defer pkg_allocator.free(output_shape);
    errdefer output.deinit();

    try floor_lean(T, input, &output);
    return output;
}

pub fn floor_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    // Compute floor(x) for each element of the tensor
    for (input.data, output.data) |in_val, *out_val| {
        if (std.math.isNan(in_val) or std.math.isInf(in_val) or in_val == 0 or in_val == @trunc(in_val)) {
            out_val.* = in_val;
        } else {
            out_val.* = std.math.floor(in_val);
        }
    }
}
