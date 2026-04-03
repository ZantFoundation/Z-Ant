const std = @import("std");
const zant = @import("../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;

/// Computes the output shape for the Exp operator.
/// Returns a slice with the same shape as the input, as Exp is an element-wise operation.
pub fn get_exp_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

/// Helper function to check if T is a supported float type.
pub fn isFloatType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Float => true,
        else => false,
    };
}
