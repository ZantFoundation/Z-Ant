const IR_zant = @import("IR_zant");

const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Computes the output shape for the GatherND operator.
/// Given data tensor shape [d_0, d_1, ..., d_{n-1}] and indices tensor shape [i_0, i_1, ..., i_{k-1}, r]
/// where r <= n, the output shape is [i_0, i_1, ..., i_{k-1}, d_r, d_{r+1}, ..., d_{n-1}]
pub fn get_gathernd_output_shape(data_shape: []const usize, indices_shape: []const usize) ![]usize {
    if (indices_shape.len == 0) return error.InvalidDimensions;

    const r = indices_shape[indices_shape.len - 1]; // Last dimension of indices
    if (r > data_shape.len) return error.InvalidDimensions;

    // Output shape: [i_0, i_1, ..., i_{k-1}, d_r, d_{r+1}, ..., d_{n-1}]
    const output_rank = indices_shape.len - 1 + data_shape.len - r;
    const output_shape = try pkg_allocator.alloc(usize, output_rank);

    // Copy indices shape (excluding last dimension)
    for (0..indices_shape.len - 1) |i| {
        output_shape[i] = indices_shape[i];
    }

    // Copy remaining data shape
    for (r..data_shape.len) |i| {
        output_shape[indices_shape.len - 1 + i - r] = data_shape[i];
    }

    return output_shape;
}
