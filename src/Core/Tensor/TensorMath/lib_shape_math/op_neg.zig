const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  neg( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn neg(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    try neg_lean(T, kernel, &flipped_kernel);
    return flipped_kernel;
}

pub fn neg_lean(comptime T: type, kernel: *Tensor(T), output_kernel: *Tensor(T)) !void {
    const kernel_dim = kernel.shape.len;
    const kernel_row = kernel.shape[kernel_dim - 2];
    const kernel_cols = kernel.shape[kernel_dim - 1];
    const matrix_dim = kernel_cols * kernel_row;

    const total_number_2DMatrices = output_kernel.size / matrix_dim;

    for (0..total_number_2DMatrices) |matix_i| {
        for (0..kernel_row) |i| {
            for (0..kernel_cols) |j| {
                output_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = kernel.data[matix_i * matrix_dim + i * kernel_cols + j];
            }
        }
    }
}

pub fn get_neg_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}
