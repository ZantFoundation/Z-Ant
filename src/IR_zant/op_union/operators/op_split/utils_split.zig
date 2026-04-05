const std = @import("std");
const zant = @import("zant");


const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_split_output_shapes(input_shape: []const usize, axis: i64, split_sizes: ?[]const usize, num_outputs: ?usize) ![][]usize {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_shape.len)) + axis;
        if (adjusted < 0) return error.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_shape.len) return error.InvalidAxis;

    const dim_size: usize = input_shape[positive_axis];
    var sizes: std.ArrayList(usize) = .empty;
    defer sizes.deinit(pkg_allocator);

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(pkg_allocator, size);
            total_size += size;
        }
        if (total_size != dim_size) return error.InvalidSplitSize;
    } else if (num_outputs) |n| {
        // Split into equal parts based on the number of outputs
        if (@mod(dim_size, n) != 0) return error.InvalidSplitSize;
        const split_size = @divTrunc(dim_size, n);
        var i: usize = 0;
        while (i < n) : (i += 1) try sizes.append(pkg_allocator, split_size);
    } else {
        // Default case: just create one output with the full size
        try sizes.append(pkg_allocator, dim_size);
    }

    // Create output shapes
    var output_shapes = try pkg_allocator.alloc([]usize, sizes.items.len);
    errdefer {
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Fill output shapes
    for (sizes.items, 0..) |split_size, i| {
        output_shapes[i] = try pkg_allocator.alloc(usize, input_shape.len);
        errdefer pkg_allocator.free(output_shapes[i]);

        @memcpy(output_shapes[i], input_shape);
        output_shapes[i][positive_axis] = split_size;
    }

    return output_shapes;
}
