const std = @import("std");
const IR_zant = @import("IR_zant");
const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub fn get_abs_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
