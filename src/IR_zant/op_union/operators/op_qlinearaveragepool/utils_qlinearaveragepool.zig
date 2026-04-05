const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

// Import existing average pool operation for shape calculation
const averagepool = @import("../op_averagePool/zant_averagePool.zig");

/// Calculate output shape for QLinearAveragePool - same as regular AveragePool
pub fn get_qlinearaveragepool_output_shape(
    input_shape: []const usize,
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: averagepool.AutoPadType,
    ceil_mode: bool,
) ![]usize {
    return averagepool.get_onnx_averagepool_output_shape(
        input_shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
}
