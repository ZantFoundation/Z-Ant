const std = @import("std");
const zant = @import("zant");

const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_identity_shape_output(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}
