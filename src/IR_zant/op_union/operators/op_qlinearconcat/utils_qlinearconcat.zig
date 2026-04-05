const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

// Import existing concatenate operation for shape calculation and core logic
const concatenate = @import("../op_concat/zant_concat.zig");

/// Calculate output shape for QLinearConcat - same as regular concatenate
pub fn get_qlinearconcat_output_shape(
    input_shapes: []const []const usize,
    axis: isize,
) ![]usize {
    return concatenate.get_concatenate_output_shape(input_shapes, axis);
}
