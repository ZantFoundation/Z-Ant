const std = @import("std");
const zant = @import("../../../zant.zig");

const pkgAllocator = zant.utils.allocator.allocator;

pub fn get_one_hot_output_shape(indices_shape: []const usize, depth: i64, axis: i64) ![]usize {
    // Normalize axis
    const rank = @as(i64, @intCast(indices_shape.len));
    const normalized_axis = if (axis < 0) axis + rank + 1 else axis;
    if (normalized_axis < 0 or normalized_axis > rank) {
        return error.InvalidAxes;
    }

    // Create output shape: rank(indices) + 1
    var output_shape = try pkgAllocator.alloc(usize, indices_shape.len + 1);
    errdefer pkgAllocator.free(output_shape);

    // Copy indices_shape and insert depth at position axis
    for (indices_shape, 0..) |dim, i| {
        if (i < normalized_axis) {
            output_shape[i] = dim;
        } else {
            output_shape[i + 1] = dim;
        }
    }
    output_shape[@intCast(normalized_axis)] = @intCast(depth);

    return output_shape;
}
