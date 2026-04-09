const zant = @import("zant");


const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_squeeze_output_shape(input_shape: []const usize, axes: ?[]const i64) ![]usize {
    const input_rank = input_shape.len;
    const input_rank_i64 = @as(i64, @intCast(input_rank));

    var squeeze_flags = try pkg_allocator.alloc(bool, input_rank);
    defer pkg_allocator.free(squeeze_flags);
    @memset(squeeze_flags, false);

    if (axes) |provided_axes| {
        // Mark the provided axes
        for (provided_axes) |axis| {
            var real_axis: usize = undefined;
            // Accepted range is [-input_rank, input_rank-1]
            if (axis < 0) {
                // Negative value means counting dimensions from the back
                if (axis < -input_rank_i64)
                    return error.AxisOutOfRange;
                real_axis = @as(usize, @intCast(input_rank_i64 + axis));
            } else {
                if (axis >= input_rank_i64)
                    return error.AxisOutOfRange;
                real_axis = @as(usize, @intCast(axis));
            }
            if (input_shape[real_axis] != 1)
                return error.InvalidAxes;
            squeeze_flags[real_axis] = true;
        }
    } else {
        // If axes is not provided, mark all dimensions of size 1
        for (input_shape, 0..) |dim, i| {
            if (dim == 1)
                squeeze_flags[i] = true;
        }
    }

    // Calculate output_shape rank
    var output_rank: usize = 0;
    for (squeeze_flags) |flag| {
        if (!flag) output_rank += 1;
    }

    // Construct output_shape
    const output_shape = try pkg_allocator.alloc(usize, output_rank);
    var j: usize = 0;
    for (input_shape, 0..) |dim, i| {
        if (!squeeze_flags[i]) {
            output_shape[j] = dim;
            j += 1;
        }
    }

    return output_shape;
}
