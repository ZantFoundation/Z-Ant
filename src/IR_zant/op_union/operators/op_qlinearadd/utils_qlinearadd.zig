const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

// Import existing addition operation for broadcasting logic
const addition = @import("../op_add/zant_add.zig");

/// Calculate output shape for QLinearAdd - same as regular broadcasting
pub fn get_qlinearadd_output_shape(
    input_shape_a: []const usize,
    input_shape_b: []const usize,
) ![]usize {
    return addition.calculate_broadcasted_shape(&pkg_allocator, input_shape_a, input_shape_b);
}
