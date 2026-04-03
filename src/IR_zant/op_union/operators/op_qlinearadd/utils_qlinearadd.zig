const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing addition operation for broadcasting logic
const addition = @import("../op_add/zant_add.zig");

/// Calculate output shape for QLinearAdd - same as regular broadcasting
pub fn get_qlinearadd_output_shape(
    input_shape_a: []const usize,
    input_shape_b: []const usize,
) ![]usize {
    return addition.calculate_broadcasted_shape(&pkg_allocator, input_shape_a, input_shape_b);
}
