const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

const get_concat_output_shape = @import("utils_concat.zig").get_concat_output_shape;

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
///     - error.EmptyTensorList
///     - error.AxisOutOfBounds
///     - error.MismatchedRank
///     - error.MismatchedShape
pub fn concat(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize) !Tensor(T) {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return error.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    // Find the maximum rank among all tensors
    var max_rank: usize = rank;
    var need_reshape = false;

    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            need_reshape = true;
            max_rank = @max(max_rank, tensor.shape.len);
        }
    }
    // ---- CHECKS
    // Update working rank to the potentially new maximum rank
    const working_rank = max_rank;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(working_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(working_rank))) {
        return error.AxisOutOfBounds;
    }

    // ---- COMPUTE
    var input_shapes = try allocator.alloc([]const usize, tensors.len);
    defer {
        for (input_shapes) |i| allocator.free(i);
        allocator.free(input_shapes);
    }

    for (tensors, 0..) |input, i| {
        // Handle negative values by using 1 as a placeholder
        var shape = try allocator.alloc(usize, input.shape.len);
        for (input.shape, 0..) |dim, j| {
            shape[j] = if (dim < 0) 1 else @intCast(dim);
        }
        input_shapes[i] = shape;
    }

    // Get output shape using the existing function
    const output_shape = try get_concat_output_shape(input_shapes, axis);
    defer allocator.free(output_shape);

    // Create the output tensor
    var output_tensor = try Tensor(T).fromShape(allocator, output_shape);

    // Use the lean version to perform the actual concatenation
    concat_lean(T, allocator, tensors, axis, &output_tensor) catch |err| {
        return err;
    };

    return output_tensor;
}

pub fn concat_lean(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize, output: *Tensor(T)) !void {
    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    // Find the maximum rank among all tensors
    var max_rank: usize = rank;
    var need_reshape = false;

    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            need_reshape = true;
            max_rank = @max(max_rank, tensor.shape.len);
        }
    }

    // Create a working copy of the tensors that we might modify
    var modified_tensors = try allocator.alloc(Tensor(T), tensors.len);
    defer {
        // Clean up any reshaped tensors we created
        if (need_reshape) {
            for (modified_tensors) |*tensor| {
                tensor.deinit();
            }
        }
        allocator.free(modified_tensors);
    }

    // Initially, just copy the references
    for (tensors, 0..) |tensor, i| {
        modified_tensors[i] = tensor;
    }

    // Handle reshaping if needed
    if (need_reshape) {
        // Reshape tensors with lower rank to match the maximum rank
        for (tensors, 0..) |tensor, i| {
            if (tensor.shape.len < max_rank) {
                // Create a new shape with added dimensions
                var new_shape = try allocator.alloc(usize, max_rank);
                defer allocator.free(new_shape);

                // Fill with 1s first
                @memset(new_shape, 1);

                // Copy original dimensions
                const offset = max_rank - tensor.shape.len;
                for (tensor.shape, 0..) |dim, j| {
                    new_shape[offset + j] = dim;
                }

                // Create a new tensor with the reshaped dimensions
                // Replace the original tensor in our working copy
                modified_tensors[i] = try Tensor(T).fromArray(allocator, tensor.data, new_shape);
            }
        }
    }

    // Update working rank to the potentially new maximum rank
    const working_rank = max_rank;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(working_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(working_rank))) {
        return error.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have matching shapes except along the concatenation axis
    for (modified_tensors) |tensor| {
        for (0..working_rank) |d| {
            if (d != concat_axis_usize and tensor.shape[d] != modified_tensors[0].shape[d]) {
                return error.MismatchedShape;
            }
        }
    }

    // Calculate the number of slices based on the concatenation axis
    var num_slices: usize = 1;
    for (0..concat_axis_usize) |d| {
        num_slices *= output.shape[d];
    }

    // Calculate the slice size (number of elements to copy per concatenation dimension)
    var slice_size: usize = 1;
    if (concat_axis_usize + 1 < working_rank) {
        for ((concat_axis_usize + 1)..working_rank) |d| {
            slice_size *= output.shape[d];
        }
    } else {
        slice_size = 1;
    }

    // Initialize the offset for copying data into output
    var offset: usize = 0;

    // Iterate over each slice
    for (0..num_slices) |slice_idx| {
        for (modified_tensors) |tensor| {
            const concat_dim = tensor.shape[concat_axis_usize];
            const copy_size = concat_dim * slice_size;

            // Calculate the start and end indices in the source tensor
            const src_start = slice_idx * concat_dim * slice_size;
            const src_end = src_start + copy_size;

            // Check bounds for the source tensor's data
            if (src_end > tensor.data.len) {
                return error.IndexOutOfBounds;
            }

            // Calculate the destination indices in output data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the output buffer
            if (dest_end > output.data.len) {
                return error.IndexOutOfBounds;
            }

            @memcpy(output.data[dest_start..dest_end], tensor.data[src_start .. src_start + copy_size]);

            // Update the offset for the next copy
            offset += copy_size;
        }
    }
}
