const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

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

pub fn get_flatten_output_shape(input_shape: []const usize, axis: isize) ![]usize {
    const rank = input_shape.len;
    const r = @as(isize, @intCast(rank));
    if (axis < -r or axis > r) {
        return TensorMathError.AxisOutOfRange;
    }

    const normalized_axis = if (axis < 0) axis + r else axis;
    const usize_axis = @as(usize, @intCast(normalized_axis));
    //calculate outer and inner dimensions
    var outer_dim: usize = 1;
    for (input_shape[0..usize_axis]) |dim| {
        outer_dim *= dim;
    }

    var inner_dim: usize = 1;
    for (input_shape[usize_axis..]) |dim| {
        inner_dim *= dim;
    }

    //create output shape
    var output_shape = try pkg_allocator.alloc(usize, 2);
    errdefer pkg_allocator.free(output_shape);

    output_shape[0] = outer_dim;
    output_shape[1] = inner_dim;

    return output_shape;
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
        return TensorMathError.InvalidInput;
    }
    if (input.shape.len > 0 and input.data.len != expected_size) {
        return TensorMathError.InvalidInput;
    }

    //TODO: verify that T is among the supported types:
    // const type_info = @typeInfo(T);
    // if (type_info != .float and type_info != .int and type_info != .bool and type_info) {
    //     return TensorMathError.InvalidDataType;
    // }
    const output_shape = try get_flatten_output_shape(input.shape, axis);
    defer pkg_allocator.free(output_shape);

    var output = try TensorOf(T).fromShape(input.allocator, output_shape);
    errdefer output.deinit();

    try flatten_lean(T, input, &output);

    return output;
}
