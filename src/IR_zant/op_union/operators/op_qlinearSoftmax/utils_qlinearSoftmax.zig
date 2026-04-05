const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

/// Calculate output shape for QLinearSoftmax - same as input shape
pub fn get_qlinearsoftmax_output_shape(input_shape: []const usize) ![]usize {
    return pkg_allocator.dupe(usize, input_shape);
}
