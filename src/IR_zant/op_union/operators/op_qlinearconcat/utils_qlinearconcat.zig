const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

// Import existing concatenate operation for shape calculation and core logic
const concatenate = @import("../op_concat/zant_concat.zig");

/// Calculate output shape for QLinearConcat - same as regular concatenate
pub fn get_qlinearconcat_output_shape(
    input_shapes: []const []const usize,
    axis: isize,
) ![]usize {
    return concatenate.get_concatenate_output_shape(input_shapes, axis);
}
