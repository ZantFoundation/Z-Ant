const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

// Squeeze - 23 : https://onnx.ai/onnx/operators/onnx__Squeeze.html
// Remove single-dimensional entries from the shape of a tensor
// Takes an input axes with a list of axes to squeeze
// If axes is not provided, all the single dimensions will be removed from the shape
// If an axis is selected with shape entry not equal to one, an error is raised

pub fn get_squeeze_output_shape(input_shape: []const usize, axes: ?[]const i64) ![]usize {
    const input_rank = input_shape.len;
    const input_rank_i64 = @as(i64, @intCast(input_rank));

    var squeeze_flags = try pkg_allocator.alloc(bool, input_rank);
    defer pkg_allocator.free(squeeze_flags);
    @memset(squeeze_flags, false);

    if (axes) |provided_axes| {
        // Mark the provided axes
        for (provided_axes) |axis| {
            var real_axis: usize = undefined;
            // Accepted range is [-input_rank, input_rank-1]
            if (axis < 0) {
                // Negative value means counting dimensions from the back
                if (axis < -input_rank_i64)
                    return TensorMathError.AxisOutOfRange;
                real_axis = @as(usize, @intCast(input_rank_i64 + axis));
            } else {
                if (axis >= input_rank_i64)
                    return TensorMathError.AxisOutOfRange;
                real_axis = @as(usize, @intCast(axis));
            }
            if (input_shape[real_axis] != 1)
                return TensorMathError.InvalidAxes;
            squeeze_flags[real_axis] = true;
        }
    } else {
        // If axes is not provided, mark all dimensions of size 1
        for (input_shape, 0..) |dim, i| {
            if (dim == 1)
                squeeze_flags[i] = true;
        }
    }

    // Calculate output_shape rank
    var output_rank: usize = 0;
    for (squeeze_flags) |flag| {
        if (!flag) output_rank += 1;
    }

    // Construct output_shape
    const output_shape = try pkg_allocator.alloc(usize, output_rank);
    var j: usize = 0;
    for (input_shape, 0..) |dim, i| {
        if (!squeeze_flags[i]) {
            output_shape[j] = dim;
            j += 1;
        }
    }

    return output_shape;
}

pub fn squeeze_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    @memcpy(output.data, input.data);
}

pub fn squeeze(comptime T: anytype, input: *Tensor(T), axes: ?[]const i64) !Tensor(T) {
    const output_shape = try get_squeeze_output_shape(input.shape, axes);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(input.allocator, output_shape);
    errdefer output.deinit();

    try squeeze_lean(T, input, &output);

    return output;
}
