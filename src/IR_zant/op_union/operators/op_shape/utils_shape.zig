const std = @import("std");
const zant = @import("zant");

const pkg_allocator = zant.utils.allocator.allocator;

/// Calculate the output shape for an ONNX Shape operation without performing the operation
pub fn get_shape_output_shape(input_shape: []const usize, start: ?i64, end: ?i64) ![]usize {
    const rank = input_shape.len;

    // Allocate output_shape (always a 1D tensor)
    var output_shape = try pkg_allocator.alloc(usize, 1);
    errdefer pkg_allocator.free(output_shape);

    // Special case for rank 0 (scalar tensor)
    if (rank == 0) {
        output_shape[0] = 0; // No dimensions to represent
        return output_shape;
    }

    // Handle start parameter
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank))));

    // Handle end parameter
    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    // Compute the output dimension
    const output_size = @max(0, end_axis - start_axis);
    output_shape[0] = @intCast(output_size);

    return output_shape;
}
