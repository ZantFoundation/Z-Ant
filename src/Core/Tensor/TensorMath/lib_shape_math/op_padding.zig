const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

/// Method to add a top&bottom padding and a left&right padding.
/// At the moment the function only supports 2 padding params, but the method
/// is already set to have different left, right, top and bottom padding values.
pub fn addPaddingAndDilation(
    comptime T: type,
    t: *Tensor(T),
    upDownPadding: usize,
    leftRightPadding: usize,
    verticalDil: usize,
    horizontalDil: usize,
) !void {

    //checks on padding dim (usize is alway >= 0)
    if (t.shape.len < 2) return TensorError.TooSmallToPadding;

    const upPadding = upDownPadding;
    const downPadding = upDownPadding;
    const leftPadding = leftRightPadding;
    const rightPadding = leftRightPadding;
    const dim = t.shape.len;

    const new_row_numb = t.shape[dim - 2] + upPadding + downPadding + verticalDil * (t.shape[dim - 2] - 1);
    const new_col_numb = t.shape[dim - 1] + leftPadding + rightPadding + horizontalDil * (t.shape[dim - 1] - 1);
    //std.debug.print("\n new_row_numb: {} new_col_numb:{}", .{ new_row_numb, new_col_numb });

    //compute new shape
    const new_shape = try t.allocator.alloc(usize, dim);
    @memcpy(new_shape, t.shape);
    new_shape[dim - 1] = new_col_numb;
    new_shape[dim - 2] = new_row_numb;

    //compute new size
    var new_total_size: usize = 1;
    for (new_shape) |size_i| {
        new_total_size *= size_i;
    }

    //alloc new tensor.data memory space to all zero
    const new_data = try t.allocator.alloc(T, new_total_size);
    @memset(new_data, 0);

    const new_matrix_dim = new_row_numb * new_col_numb;
    const total_number_2DMatrices = new_total_size / new_matrix_dim;
    const old_matrix_dim = t.shape[dim - 2] * t.shape[dim - 1];
    const old_total_number_2DMatrices = t.size / old_matrix_dim; //just for check assertion
    std.debug.assert(total_number_2DMatrices == old_total_number_2DMatrices);

    for (0..total_number_2DMatrices) |matix_i| {
        const num_elem_prec_new_matr = matix_i * new_matrix_dim;
        const num_elem_prec_old_matr = matix_i * old_matrix_dim;
        var i = upPadding;
        var old_row: usize = 0;
        while (i < new_row_numb - downPadding) : (i += (1 + verticalDil)) {
            var j = leftPadding;
            var old_col: usize = 0;
            while (j < new_col_numb - rightPadding) : (j += (1 + horizontalDil)) {
                const idx_new_matr = num_elem_prec_new_matr + i * new_col_numb + j;
                const idx_old_matr = num_elem_prec_old_matr + old_row * (t.shape[dim - 1]) + old_col;
                new_data[idx_new_matr] = t.data[idx_old_matr];
                old_col += 1;
            }
            old_row += 1;
        }
    }

    //free all old attributes and setting new ones
    t.allocator.free(t.data);
    t.allocator.free(t.shape);

    t.shape = new_shape;
    t.data = new_data;
    t.size = new_total_size;
}

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  flip( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn flip(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    const kernel_dim = kernel.shape.len;
    const kernel_row = kernel.shape[kernel_dim - 2];
    const kernel_cols = kernel.shape[kernel_dim - 1];
    const matrix_dim = kernel_cols * kernel_row;

    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    const total_number_2DMatrices = flipped_kernel.size / matrix_dim;

    for (0..total_number_2DMatrices) |matix_i| {
        for (0..kernel_row) |i| {
            for (0..kernel_cols) |j| {
                flipped_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = kernel.data[matix_i * matrix_dim + i * kernel_cols + j];
            }
        }
    }

    return flipped_kernel;
}
