const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Calculate output shape for GlobalAveragePool operation
/// For GlobalAveragePool, output shape is (N, C, 1, 1, ...) where all spatial dimensions become 1
pub fn get_global_average_pool_output_shape(input_shape: []const usize) ![]usize {
    if (input_shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);

    // First two dimensions (batch_size, channels) remain the same
    output_shape[0] = input_shape[0]; // N (batch size)
    output_shape[1] = input_shape[1]; // C (channels)

    // All spatial dimensions become 1
    for (2..input_shape.len) |i| {
        output_shape[i] = 1;
    }

    return output_shape;
}
