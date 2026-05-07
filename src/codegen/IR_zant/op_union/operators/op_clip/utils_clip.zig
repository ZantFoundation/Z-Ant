const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Returns the shape of the output tensor for the clip operation.
/// For clip operation, the output shape is identical to the input shape.
pub fn get_clip_output_shape(comptime T: type, inputTensor: *const Tensor(T), minTensor: ?*const Tensor(T), maxTensor: ?*const Tensor(T)) ![]usize {
    const allocator = inputTensor.allocator;
    _ = minTensor;
    _ = maxTensor;
    const shape = try allocator.alloc(usize, inputTensor.shape.len);
    @memcpy(shape, inputTensor.shape);
    return shape;
}

pub const clip_quantized_lean = @import("../../fused/fused_dequant_clip_quant/zant_clip_quantized.zig").clip_quantized_lean;
