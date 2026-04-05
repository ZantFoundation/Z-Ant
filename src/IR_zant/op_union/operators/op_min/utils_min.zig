const std = @import("std");
const zant = @import("../../../zant.zig");

const pkg_allocator = zant.utils.allocator.allocator;

/// Computes the output shape for the Min operator.
/// Min is an element-wise operation, so output shape is the broadcast shape of all inputs.
pub fn get_min_output_shape(input_shapes: []const []const usize) ![]usize {
    if (input_shapes.len == 0) return error.InvalidDimensions;
    if (input_shapes.len == 1) {
        const output_shape = try pkg_allocator.alloc(usize, input_shapes[0].len);
        @memcpy(output_shape, input_shapes[0]);
        return output_shape;
    }

    // For multiple inputs, we need to compute broadcast shape
    // For simplicity, assume all inputs have the same shape (most common case)
    // TODO: Implement proper broadcasting logic
    const output_shape = try pkg_allocator.alloc(usize, input_shapes[0].len);
    @memcpy(output_shape, input_shapes[0]);
    return output_shape;
}
