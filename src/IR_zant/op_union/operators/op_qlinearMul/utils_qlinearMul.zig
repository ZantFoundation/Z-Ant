const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

// Import existing multiplication operation for broadcasting logic
const multiplication = @import("../op_mul/zant_mul.zig");

/// Calculate output shape for QLinearMul - same as regular Mul (uses broadcasting)
pub fn get_qlinearmul_output_shape(
    a_shape: []const usize,
    b_shape: []const usize,
) ![]usize {
    return multiplication.utils.get_mul_output_shape(a_shape, b_shape);
}
