const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;

const pkg_allocator = zant.utils.allocator.allocator;

const get_flatten_output_shape = @import("utils_flatten.zig").get_flatten_output_shape;

// Increase comptime evaluation limit for complex flatten operations
comptime {
    @setEvalBranchQuota(1000000);
}

// Helper to raise quota when evaluating Tensor(T) as a type
inline fn TensorOf(comptime T: type) type {
    comptime {
        @setEvalBranchQuota(1000000);
    }
    return Tensor(T);
}

pub fn flatten_lean(comptime T: type, input_any: anytype, output_any: anytype) !void {
    @setEvalBranchQuota(1000000);
    const TensorT = TensorOf(T);
    const input: *TensorT = input_any;
    const output: *TensorT = output_any;
    @memcpy(output.data, input.data);
}

pub fn flatten(comptime T: type, input: *TensorOf(T), axis: isize) !TensorOf(T) {
    //validate input
    var expected_size: usize = 1;
    for (input.shape) |dim| {
        expected_size = try std.math.mul(usize, expected_size, dim);
    }
    if (input.shape.len == 0 and input.data.len != 1) {
        return error.InvalidInput;
    }
    if (input.shape.len > 0 and input.data.len != expected_size) {
        return error.InvalidInput;
    }

    const output_shape = try get_flatten_output_shape(input.shape, axis);
    defer pkg_allocator.free(output_shape);

    var output = try TensorOf(T).fromShape(input.allocator, output_shape);
    errdefer output.deinit();

    try flatten_lean(T, input, &output);

    return output;
}
