const std = @import("std");
const zant = @import("../../../../zant.zig");

pub inline fn get_tanh_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try zant.utils.allocator.allocator.alloc(usize, input_shape.len);
    errdefer zant.utils.allocator.allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
