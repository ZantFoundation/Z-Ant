const IR_zant = @import("IR_zant");


const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub fn get_flatten_output_shape(input_shape: []const usize, axis: isize) ![]usize {
    const rank = input_shape.len;
    const r = @as(isize, @intCast(rank));
    if (axis < -r or axis > r) {
        return error.AxisOutOfRange;
    }

    const normalized_axis = if (axis < 0) axis + r else axis;
    const usize_axis = @as(usize, @intCast(normalized_axis));
    //calculate outer and inner dimensions
    var outer_dim: usize = 1;
    for (input_shape[0..usize_axis]) |dim| {
        outer_dim *= dim;
    }

    var inner_dim: usize = 1;
    for (input_shape[usize_axis..]) |dim| {
        inner_dim *= dim;
    }

    //create output shape
    var output_shape = try pkg_allocator.alloc(usize, 2);
    errdefer pkg_allocator.free(output_shape);

    output_shape[0] = outer_dim;
    output_shape[1] = inner_dim;

    return output_shape;
}
