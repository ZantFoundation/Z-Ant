const IR_zant = @import("IR_zant");

const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Computes the output shape for the TopK operator.
/// Returns two shapes: one for values and one for indices.
pub fn get_topk_output_shape(
    input_shape: []const usize,
    k: usize,
    axis: i64,
) !struct { values_shape: []usize, indices_shape: []usize } {
    if (input_shape.len == 0) {
        return error.InvalidInput;
    }

    // Normalize axis
    const normalized_axis = if (axis < 0)
        @as(usize, @intCast(@as(i64, @intCast(input_shape.len)) + axis))
    else
        @as(usize, @intCast(axis));

    if (normalized_axis >= input_shape.len) {
        return error.InvalidInput;
    }

    // Check if k is valid
    if (k > input_shape[normalized_axis]) {
        return error.InvalidInput;
    }

    // Create output shapes
    const values_shape = try pkg_allocator.alloc(usize, input_shape.len);
    const indices_shape = try pkg_allocator.alloc(usize, input_shape.len);

    @memcpy(values_shape, input_shape);
    @memcpy(indices_shape, input_shape);

    values_shape[normalized_axis] = k;
    indices_shape[normalized_axis] = k;

    return .{ .values_shape = values_shape, .indices_shape = indices_shape };
}
