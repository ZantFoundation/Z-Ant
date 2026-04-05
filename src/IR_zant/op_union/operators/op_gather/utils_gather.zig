const std = @import("std");
const zant = @import("../../../../zant.zig");


const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_gather_output_shape(input_shape: []const usize, indices_shape: []const usize, selected_axis: isize) ![]usize {

    // Scalar data tensor is not allowed
    if (input_shape.len == 0) {
        return error.InvalidRank;
    }

    // Validate that the axis is within the tensor's dimensions
    const number_dimensions: isize = @intCast(input_shape.len);
    if (selected_axis >= number_dimensions or selected_axis < -number_dimensions) {
        return error.InvalidAxis;
    }

    // If axis is negative, convert it to a positive index
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // Calculate the shape of the output tensor:
    // [input_shape[0..axis], indices_shape..., input_shape[axis+1..]]
    const output_shape_len = input_shape.len + indices_shape.len - 1;
    const output_shape = try pkg_allocator.alloc(usize, output_shape_len);
    errdefer pkg_allocator.free(output_shape);

    // Copy the dimensions before the axis
    for (0..axis) |i| {
        output_shape[i] = input_shape[i];
    }

    // Copy indices shape
    var indices_idx: usize = 0;
    while (indices_idx < indices_shape.len) : (indices_idx += 1) {
        output_shape[axis + indices_idx] = indices_shape[indices_idx];
    }

    // Copy the dimensions after the axis
    for (axis + 1..input_shape.len) |i| {
        output_shape[axis + indices_shape.len + (i - axis - 1)] = input_shape[i];
    }

    return output_shape;
}
