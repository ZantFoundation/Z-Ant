const std = @import("std");
const zant = @import("zant");
const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_log_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

// Check whether the tensor type is supported or not
pub fn isLogSupportedType(comptime T: type) bool {
    return switch (T) {
        f16, f32, f64 => true,
        else => false,
    };
}
