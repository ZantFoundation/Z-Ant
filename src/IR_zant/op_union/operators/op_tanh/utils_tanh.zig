const std = @import("std");
const IR_zant = @import("IR_zant");

pub inline fn get_tanh_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try IR_zant.pkg_allocator.allocator.alloc(usize, input_shape.len);
    errdefer IR_zant.pkg_allocator.allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
