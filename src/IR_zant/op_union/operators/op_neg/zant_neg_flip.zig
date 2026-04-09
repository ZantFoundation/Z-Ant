const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  flip_matrix( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn flip_matrix(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    try flip_matrix_lean(T, kernel, &flipped_kernel);
    return flipped_kernel;
}

pub fn flip_matrix_lean(comptime T: type, kernel: *Tensor(T), output_kernel: *Tensor(T)) !void {
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
