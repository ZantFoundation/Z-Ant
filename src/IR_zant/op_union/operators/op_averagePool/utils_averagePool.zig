const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub const AutoPadType = @import("zant_averagePool.zig").AutoPadType;

/// Calculate output shape for ONNX AveragePool operation
/// This is where ceil_mode is actually used
pub fn get_average_pool_output_shape(
    input_shape: []const usize,
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
) ![]usize {
    if (input_shape.len < 3) return error.InvalidInputRank;

    const spatial_dims = input_shape.len - 2;
    if (kernel_shape.len != spatial_dims) return error.KernelShapeMismatch;
    if (strides.len != spatial_dims) return error.StridesMismatch;
    if (dilations.len != spatial_dims) return error.DilationsMismatch;

    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    // Copy batch and channel dimensions
    output_shape[0] = input_shape[0]; // batch
    output_shape[1] = input_shape[1]; // channels

    // Calculate spatial dimensions using exact ONNX formulas
    for (0..spatial_dims) |i| {
        const input_size = input_shape[2 + i];
        const kernel_size = kernel_shape[i];
        const stride = strides[i];
        const dilation = dilations[i];

        var output_size: usize = 0;

        switch (auto_pad) {
            .NOTSET => {
                var pad_begin: usize = 0;
                var pad_end: usize = 0;
                if (i < pads.len) pad_begin = pads[i];
                if (i + spatial_dims < pads.len) pad_end = pads[i + spatial_dims];

                const effective_kernel_size = dilation * (kernel_size - 1) + 1;
                const total_pad = pad_begin + pad_end;
                const padded_input_size = input_size + total_pad;

                if (padded_input_size >= effective_kernel_size) {
                    const numerator = padded_input_size - effective_kernel_size;
                    if (ceil_mode) {
                        output_size = (numerator + stride) / stride;
                    } else {
                        output_size = (numerator / stride) + 1;
                    }
                } else {
                    output_size = 1;
                }
            },
            .VALID => {
                const effective_kernel_size = (kernel_size - 1) * dilation + 1;
                if (input_size >= effective_kernel_size) {
                    const numerator = input_size - effective_kernel_size;
                    if (ceil_mode) {
                        output_size = (numerator + stride) / stride + 1;
                    } else {
                        output_size = (numerator / stride) + 1;
                    }
                } else {
                    output_size = 1;
                }
            },
            .SAME_UPPER, .SAME_LOWER => {
                if (ceil_mode) {
                    output_size = (input_size + stride - 1) / stride;
                } else {
                    output_size = if (input_size > 0) ((input_size - 1) / stride) + 1 else 1;
                }
            },
        }

        output_shape[2 + i] = output_size;
    }

    return output_shape;
}
