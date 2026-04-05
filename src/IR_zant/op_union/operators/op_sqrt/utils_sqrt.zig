const std = @import("std");
const zant = @import("zant");
const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_sqrt_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}
