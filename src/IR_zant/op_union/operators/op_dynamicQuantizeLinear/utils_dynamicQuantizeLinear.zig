const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;

const pkg_allocator = zant.utils.allocator.allocator;

/// Get output shapes for DynamicQuantizeLinear operation
pub fn get_dynamicQuantizeLinear_output_shape(input_shape: []const usize) ![][]usize {
    const allocator = pkg_allocator;

    // DynamicQuantizeLinear produces 3 outputs:
    // 1. quantized tensor (same shape as input)
    // 2. scale (scalar)
    // 3. zero_point (scalar)

    const output_shapes = try allocator.alloc([]usize, 3);

    // Output 0: quantized tensor - same shape as input
    output_shapes[0] = try allocator.dupe(usize, input_shape);

    // Output 1: scale - scalar (empty shape)
    output_shapes[1] = try allocator.alloc(usize, 0);

    // Output 2: zero_point - scalar (empty shape)
    output_shapes[2] = try allocator.alloc(usize, 0);

    return output_shapes;
}
