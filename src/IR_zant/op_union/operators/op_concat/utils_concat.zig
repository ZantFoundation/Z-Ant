const std = @import("std");
const IR_zant = @import("IR_zant");


const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub fn get_concat_output_shape(tensors: []const []const usize, axis: isize) ![]usize {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return error.EmptyTensorList;

    // Find the maximum rank among all tensors
    var max_rank: usize = 0;
    for (tensors) |tensor| {
        max_rank = @max(max_rank, tensor.len);
    }

    // Handle negative axis values (numpy style)
    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(max_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(max_rank))) {
        std.log.debug("\n axis out of bounds: {} (max_rank: {})", .{ concat_axis, max_rank });
        return error.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

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
    }

    // Validate that all tensors have matching shapes except along the concatenation axis
    for (broadcasted_shapes, 0..) |shape, i| {
        for (0..max_rank) |d| {
            if (d != concat_axis_usize and shape[d] != broadcasted_shapes[0][d]) {
                std.log.debug("\n Shape mismatch at dim {}: shape[{}][{}] = {} != shape[0][{}] = {}", .{ d, i, d, shape[d], d, broadcasted_shapes[0][d] });
                return error.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer {
        std.log.debug("\n Error occurred, freeing new_shape", .{});
        pkg_allocator.free(new_shape);
    }

    for (0..max_rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (broadcasted_shapes) |shape| {
                sum += shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = broadcasted_shapes[0][d];
        }
    }

    // Clean up broadcasted shapes
    for (broadcasted_shapes) |shape| {
        pkg_allocator.free(shape);
    }
    pkg_allocator.free(broadcasted_shapes);

    return new_shape;
}
