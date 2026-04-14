const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Calculate output shape for QLinearSoftmax - same as input shape
pub fn get_qlinearsoftmax_output_shape(input_shape: []const usize) ![]usize {
    return pkg_allocator.dupe(usize, input_shape);
}
