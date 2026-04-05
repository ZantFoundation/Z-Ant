const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub const AutoPadType = @import("zant_maxPool.zig").AutoPadType;

/// Helper function to calculate output shape for MaxPool
pub fn get_max_pool_output_shape(
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

    // Copy batch and channel dimensions
    output_shape[0] = input_shape[0]; // batch
    output_shape[1] = input_shape[1]; // channels

    // Calculate spatial dimensions
    for (0..spatial_dims) |i| {
        const input_size = input_shape[2 + i];
        const kernel_size = kernel_shape[i];
        const stride = strides[i];
        const dilation = dilations[i];

        var pad_begin: usize = 0;
        var pad_end: usize = 0;

        switch (auto_pad) {
            .NOTSET => {
                if (i < pads.len) pad_begin = pads[i];
                if (i + spatial_dims < pads.len) pad_end = pads[i + spatial_dims];
            },
            .VALID => {
                // No padding
            },
            .SAME_UPPER, .SAME_LOWER => {
                // Calculate padding for SAME mode
                const effective_kernel_size = (kernel_size - 1) * dilation + 1;
                const output_size = (input_size + stride - 1) / stride; // Ceiling division
                const total_pad = @max(0, @as(isize, @intCast((output_size - 1) * stride + effective_kernel_size)) - @as(isize, @intCast(input_size)));

                if (auto_pad == .SAME_UPPER) {
                    pad_begin = @as(usize, @intCast(total_pad / 2));
                    pad_end = @as(usize, @intCast(total_pad - @as(isize, @intCast(pad_begin))));
                } else {
                    pad_end = @as(usize, @intCast(total_pad / 2));
                    pad_begin = @as(usize, @intCast(total_pad - @as(isize, @intCast(pad_end))));
                }
            },
        }

        const effective_kernel_size = (kernel_size - 1) * dilation + 1;
        const padded_input_size = input_size + pad_begin + pad_end;

        if (padded_input_size < effective_kernel_size) {
            output_shape[2 + i] = 0;
        } else {
            const numerator = padded_input_size - effective_kernel_size;
            if (ceil_mode) {
                output_shape[2 + i] = (numerator + stride) / stride; // Ceiling division
            } else {
                output_shape[2 + i] = numerator / stride + 1;
            }
        }
    }

    return output_shape;
}
