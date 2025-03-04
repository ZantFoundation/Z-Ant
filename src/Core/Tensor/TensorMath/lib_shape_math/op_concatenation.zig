const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

/// Concatenates a list of tensors into a single tensor along the specified axis.
/// All input tensors must have the same shape, except for the size of the concatenation axis.
///
/// Parameters:
///     allocator - The memory allocator to use for the new tensor.
///     tensors - An array of tensors to concatenate.
///     axis - The axis along which to concatenate. Negative values count dimensions from the back.
///
/// Returns:
///     A new tensor resulting from concatenation.
///
/// Errors:
///     - TensorError.EmptyTensorList
///     - TensorError.AxisOutOfBounds
///     - TensorError.MismatchedRank
///     - TensorError.MismatchedShape
pub fn concatenate(comptime T: type, allocator: *std.mem.Allocator, tensors: []Tensor(T), axis: isize) !Tensor(T) {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            return TensorError.MismatchedRank;
        }
        for (0..rank) |d| {
            if (d != concat_axis_usize and tensor.shape[d] != tensors[0].shape[d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try allocator.alloc(usize, rank);
    for (0..rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (tensors) |tensor| {
                sum += tensor.shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = tensors[0].shape[d];
        }
    }

    // Calculate the total number of elements in the new tensor
    var total_size: usize = 1;
    for (new_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for the new tensor's data
    var new_data = try allocator.alloc(T, total_size);

    // Calculate the number of slices based on the concatenation axis
    var num_slices: usize = 1;
    for (0..concat_axis_usize) |d| {
        num_slices *= new_shape[d];
    }

    // Calculate the slice size (number of elements to copy per concatenation dimension)
    var slice_size: usize = 1;
    if (concat_axis_usize + 1 < rank) {
        for ((concat_axis_usize + 1)..rank) |d| {
            slice_size *= new_shape[d];
        }
    } else {
        slice_size = 1;
    }

    // Initialize the offset for copying data into new_data
    var offset: usize = 0;

    // Iterate over each slice
    for (0..num_slices) |slice_idx| {
        for (tensors, 0..) |tensor, tensor_idx| {
            const concat_dim = tensor.shape[concat_axis_usize];
            const copy_size = concat_dim * slice_size;

            std.debug.print("\n  Copying Tensor {}: slice_idx={} concat_dim={} slice_size={} copy_size={} to new_data[{}..{}]", .{ tensor_idx, slice_idx, concat_dim, slice_size, copy_size, offset, offset + copy_size });

            // Calculate the start and end indices in the source tensor
            const src_start = slice_idx * concat_dim * slice_size;
            const src_end = src_start + copy_size;

            // Check bounds for the source tensor's data
            if (src_end > tensor.data.len) {
                std.debug.print("\n  Out of bounds error for tensor idx:{} src_end:{} tensor.data.len:{}", .{ tensor_idx, src_end, tensor.data.len });
                return TensorError.IndexOutOfBounds;
            }

            // Calculate the destination indices in new_data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the new_data buffer
            if (dest_end > new_data.len) {
                std.debug.print("\n  Out of bounds error for new_data dest_end:{} new_data.len:{}", .{ dest_end, new_data.len });
                return TensorError.IndexOutOfBounds;
            }

            @memcpy(new_data[dest_start..dest_end], tensor.data[src_start..src_end]);

            // Update the offset for the next copy
            offset += copy_size;
        }
    }

    // Return the concatenated tensor
    return Tensor(T){
        .data = new_data,
        .size = total_size,
        .shape = new_shape,
        .allocator = allocator,
    };
}

pub fn get_concatenate_output_shape(tensors: []const []const usize, axis: isize) ![]usize {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].len;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (tensors) |tensor_shape| {
        if (tensor_shape.len != rank) {
            return TensorError.MismatchedRank;
        }
        for (0..rank) |d| {
            if (d != concat_axis_usize and tensor_shape[d] != tensors[0][d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try pkg_allocator.alloc(usize, rank);
    errdefer pkg_allocator.free(new_shape);

    for (0..rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (tensors) |tensor_shape| {
                sum += tensor_shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = tensors[0][d];
        }
    }

    return new_shape;
}
