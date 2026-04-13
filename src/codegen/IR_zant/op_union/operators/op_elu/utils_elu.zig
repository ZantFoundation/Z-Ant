const IR_zant = @import("IR_zant");
const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Computes the output shape for the Elu operator.
/// Returns a slice with the same shape as the input, as Elu is an element-wise operation.
pub fn get_elu_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

/// Helper function to check if T is a supported float type.
pub inline fn isFloatType(comptime T: type) bool {
    return T == f32 or T == f64 or T == f16 or T == f128;
}
