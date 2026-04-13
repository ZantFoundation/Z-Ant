const IR_zant = @import("IR_zant");

const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Calculate output shape for Pad operation
pub fn get_pad_output_shape(
    input_shape: []const usize,
    pads: []const i64,
    axes: ?[]const i64,
) ![]usize {
    var output_shape = try pkg_allocator.dupe(usize, input_shape);

    const num_axes = if (axes) |a| a.len else input_shape.len;
    if (pads.len < num_axes * 2) return error.InvalidDimensions;

    for (0..num_axes) |i| {
        const axis_idx = if (axes) |a| @as(usize, @intCast(a[i])) else i;
        if (axis_idx >= input_shape.len) return error.InvalidDimensions;

        const pad_begin = @as(usize, @intCast(@max(0, pads[i])));
        const pad_end = @as(usize, @intCast(@max(0, pads[i + num_axes])));
        output_shape[axis_idx] = input_shape[axis_idx] + pad_begin + pad_end;
    }

    return output_shape;
}
