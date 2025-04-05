const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

pub fn lean_concatenate(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize, output: *Tensor(T)) !void {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

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
                if (tensor.owns_memory) {
                    tensor.deinit();
                }
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
                var reshaped_tensor = try Tensor(T).init(allocator);
                errdefer reshaped_tensor.deinit();
                try reshaped_tensor.fill(tensor.data, new_shape);
                reshaped_tensor.owns_memory = true;

                // Replace the original tensor in our working copy
                modified_tensors[i] = reshaped_tensor;
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
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have matching shapes except along the concatenation axis
    for (modified_tensors) |tensor| {
        for (0..working_rank) |d| {
            if (d != concat_axis_usize and tensor.shape[d] != modified_tensors[0].shape[d]) {
                return TensorError.MismatchedShape;
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
                return TensorError.IndexOutOfBounds;
            }

            // Calculate the destination indices in output data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the output buffer
            if (dest_end > output.data.len) {
                return TensorError.IndexOutOfBounds;
            }

            @memcpy(output.data[dest_start..dest_end], tensor.data[src_start .. src_start + copy_size]);

            // Update the offset for the next copy
            offset += copy_size;
        }
    }
}

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
pub fn concatenate(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize) !Tensor(T) {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

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

    // Update working rank to the potentially new maximum rank
    const working_rank = max_rank;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(working_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(working_rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Calculate the new shape after concatenation
    var new_shape: []usize = undefined;
    var output_data: []T = undefined;
    var shape_allocated = false;
    var data_allocated = false;

    // Allocate the shape
    new_shape = allocator.alloc(usize, working_rank) catch |err| {
        return err;
    };
    shape_allocated = true;
    errdefer {
        if (shape_allocated) {
            allocator.free(new_shape);
        }
    }

    // Initialize with the shape of the first tensor (potentially reshaped)
    if (tensors[0].shape.len < working_rank) {
        // Fill with 1s first
        @memset(new_shape, 1);

        // Copy original dimensions
        const offset = working_rank - tensors[0].shape.len;
        for (tensors[0].shape, 0..) |dim, j| {
            new_shape[offset + j] = dim;
        }
    } else {
        for (0..working_rank) |d| {
            new_shape[d] = tensors[0].shape[d];
        }
    }

    // Calculate the sum along the concatenation axis
    var sum: usize = 0;
    for (tensors) |tensor| {
        if (tensor.shape.len < working_rank) {
            // For tensors with lower rank, we need to calculate the effective dimension
            const offset = working_rank - tensor.shape.len;
            if (concat_axis_usize >= offset) {
                sum += tensor.shape[concat_axis_usize - offset];
            } else {
                sum += 1; // Implicit dimension size is 1
            }
        } else {
            sum += tensor.shape[concat_axis_usize];
        }
    }
    new_shape[concat_axis_usize] = sum;

    // Calculate the total number of elements in the new tensor
    var total_size: usize = 1;
    for (new_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for the output tensor's data
    output_data = allocator.alloc(T, total_size) catch |err| {
        allocator.free(new_shape);
        return err;
    };
    data_allocated = true;
    errdefer {
        if (data_allocated) {
            allocator.free(output_data);
        }
    }

    // Create the output tensor
    var output_tensor = Tensor(T){
        .data = output_data,
        .size = total_size,
        .shape = new_shape,
        .allocator = allocator,
        .owns_memory = true,
    };

    // Use the lean version to perform the actual concatenation
    lean_concatenate(T, allocator, tensors, axis, &output_tensor) catch |err| {
        // Since output_tensor owns the memory, we don't need to manually free it
        // The caller will handle the error and the memory will be properly freed
        return err;
    };

    return output_tensor;
}

pub fn get_concatenate_output_shape(tensors: []const []const usize, axis: isize) ![]usize {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;
    // std.debug.print("\n[DEBUG] get_concatenate_output_shape - Starting concatenation", .{});
    // std.debug.print("\n[DEBUG] tensors: {any}", .{tensors});
    // std.debug.print("\n[DEBUG] axis: {d}", .{axis});

    // Find the maximum rank among all tensors
    var max_rank: usize = 0;
    for (tensors) |tensor| {
        max_rank = @max(max_rank, tensor.len);
    }
    // std.debug.print("\n[DEBUG] max_rank: {}", .{max_rank});

    // Handle negative axis values (numpy style)
    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(max_rank));
        // std.debug.print("\n[DEBUG] normalized negative axis to: {}", .{concat_axis});
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(max_rank))) {
        std.debug.print("\n[DEBUG] axis out of bounds: {} (max_rank: {})", .{ concat_axis, max_rank });
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));
    // std.debug.print("\n[DEBUG] concat_axis_usize: {}", .{concat_axis_usize});

    // Create broadcasted shapes for all tensors
    var broadcasted_shapes = try pkg_allocator.alloc([]usize, tensors.len);
    errdefer {
        for (broadcasted_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(broadcasted_shapes);
    }

    // First, create broadcasted shapes
    for (tensors, 0..) |tensor, i| {
        broadcasted_shapes[i] = try pkg_allocator.alloc(usize, max_rank);
        // Fill with 1s first
        @memset(broadcasted_shapes[i], 1);

        if (tensor.len < max_rank) {
            // For tensors with lower rank, broadcast to match the target shape
            const target_shape = if (i == 0) tensors[1] else tensors[0];
            for (0..max_rank) |d| {
                if (d != concat_axis_usize) {
                    broadcasted_shapes[i][d] = target_shape[d];
                } else {
                    // For the concatenation axis, use the original value
                    const offset = max_rank - tensor.len;
                    broadcasted_shapes[i][d] = if (d >= offset) tensor[d - offset] else 1;
                }
            }
        } else {
            // For higher rank tensors, copy dimensions
            for (tensor, 0..) |dim, j| {
                broadcasted_shapes[i][j] = dim;
            }
        }
        // std.debug.print("\n[DEBUG] Broadcasted shape[{}]: {any}", .{ i, broadcasted_shapes[i] });
    }

    // Validate that all tensors have matching shapes except along the concatenation axis
    for (broadcasted_shapes, 0..) |shape, i| {
        for (0..max_rank) |d| {
            if (d != concat_axis_usize and shape[d] != broadcasted_shapes[0][d]) {
                std.debug.print("\n[DEBUG] Shape mismatch at dim {}: shape[{}][{}] = {} != shape[0][{}] = {}", .{ d, i, d, shape[d], d, broadcasted_shapes[0][d] });
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    // std.debug.print("\n[DEBUG] Allocating new shape array of size {}", .{max_rank});
    var new_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer {
        std.debug.print("\n[DEBUG] Error occurred, freeing new_shape", .{});
        pkg_allocator.free(new_shape);
    }

    for (0..max_rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (broadcasted_shapes) |shape| {
                sum += shape[d];
            }
            new_shape[d] = sum;
            // std.debug.print("\n[DEBUG] Concatenation dimension {}: sum = {}", .{ d, sum });
        } else {
            new_shape[d] = broadcasted_shapes[0][d];
            // std.debug.print("\n[DEBUG] Non-concatenation dimension {}: {}", .{ d, new_shape[d] });
        }
    }

    // Clean up broadcasted shapes
    for (broadcasted_shapes) |shape| {
        pkg_allocator.free(shape);
    }
    pkg_allocator.free(broadcasted_shapes);

    // std.debug.print("\n[DEBUG] Final output shape: {any}", .{new_shape});
    return new_shape;
}
