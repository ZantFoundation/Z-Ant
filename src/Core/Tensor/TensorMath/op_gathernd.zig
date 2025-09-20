const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- GATHERND OPERATOR ---------------------

/// Computes the output shape for the GatherND operator.
/// Given data tensor shape [d_0, d_1, ..., d_{n-1}] and indices tensor shape [i_0, i_1, ..., i_{k-1}, r]
/// where r <= n, the output shape is [i_0, i_1, ..., i_{k-1}, d_r, d_{r+1}, ..., d_{n-1}]
pub fn get_gathernd_output_shape(data_shape: []const usize, indices_shape: []const usize) ![]usize {
    if (indices_shape.len == 0) return TensorMathError.InvalidDimensions;

    const r = indices_shape[indices_shape.len - 1]; // Last dimension of indices
    if (r > data_shape.len) return TensorMathError.InvalidDimensions;

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

/// Applies the GatherND function, allocating a new output tensor.
pub fn gathernd(comptime T: anytype, data: *Tensor(T), indices: *Tensor(i64)) !Tensor(T) {
    const output_shape = try get_gathernd_output_shape(data.shape, indices.shape);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(data.allocator, output_shape);
    errdefer output.deinit();

    try gathernd_lean(T, data, indices, &output);
    return output;
}

/// Lean version that computes GatherND in-place using a pre-allocated output tensor.
pub fn gathernd_lean(comptime T: anytype, data: *Tensor(T), indices: *Tensor(i64), output: *Tensor(T)) !void {
    if (indices.shape.len == 0) return TensorMathError.InvalidDimensions;

    const r = indices.shape[indices.shape.len - 1]; // Last dimension of indices
    if (r > data.shape.len) return TensorMathError.InvalidDimensions;

    // Calculate strides for data tensor
    var data_strides = try pkg_allocator.alloc(usize, data.shape.len);
    defer pkg_allocator.free(data_strides);

    data_strides[data.shape.len - 1] = 1;
    if (data.shape.len > 1) {
        var i = data.shape.len - 2;
        while (true) {
            data_strides[i] = data_strides[i + 1] * data.shape[i + 1];
            if (i == 0) break;
            i -= 1;
        }
    }

    // Calculate how many indices we have (product of all dimensions except last)
    var num_indices: usize = 1;
    for (0..indices.shape.len - 1) |i| {
        num_indices *= indices.shape[i];
    }

    // Calculate size of each output slice
    var slice_size: usize = 1;
    for (r..data.shape.len) |i| {
        slice_size *= data.shape[i];
    }

    // For each index tuple
    for (0..num_indices) |idx_group| {
        // Extract the multi-dimensional index
        var index_offset: usize = 0;

        for (0..r) |dim| {
            const indices_idx = idx_group * r + dim;
            if (indices_idx >= indices.size) continue;

            var index_val = indices.data[indices_idx];

            // Handle negative indices
            if (index_val < 0) {
                index_val += @as(i64, @intCast(data.shape[dim]));
            }

            // Bounds check
            if (index_val < 0 or index_val >= @as(i64, @intCast(data.shape[dim]))) {
                return TensorMathError.IndexOutOfBounds;
            }

            index_offset += @as(usize, @intCast(index_val)) * data_strides[dim];
        }

        // Copy slice from data to output
        const output_offset = idx_group * slice_size;
        for (0..slice_size) |i| {
            if (output_offset + i >= output.size or index_offset + i >= data.size) {
                return TensorMathError.IndexOutOfBounds;
            }
            output.data[output_offset + i] = data.data[index_offset + i];
        }
    }
}
