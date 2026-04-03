const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

pub const AutoPadType = enum {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

// Lean ONNX AveragePool implementation - assumes output tensor is pre-allocated with correct shape
/// ceil_mode is not needed here since output shape is already calculated
pub fn lean_onnx_averagepool(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize, // IGNORED when auto_pad != NOTSET
    auto_pad: AutoPadType,
    count_include_pad: bool,
) !void {
    const input_rank = input.shape.len;
    const output_rank = output.shape.len;

    if (input_rank != output_rank) return error.InputOutputDifferentRank;
    if (input_rank < 3) return error.InvalidInputRank;

    const spatial_dims = input_rank - 2;
    if (kernel_shape.len != spatial_dims) return error.KernelShapeMismatch;
    if (strides.len != spatial_dims) return error.StridesMismatch;
    if (dilations.len != spatial_dims) return error.DilationsMismatch;

    const batch_size = input.shape[0];
    const channels = input.shape[1];

    // Calculate effective padding using ONNX specification formulas
    var effective_pads = try pkg_allocator.alloc(isize, spatial_dims * 2);
    defer pkg_allocator.free(effective_pads);

    switch (auto_pad) {
        .NOTSET => {
            // Use explicit padding: pads format [x1_begin, x2_begin…x1_end, x2_end,…]
            for (0..spatial_dims) |i| {
                effective_pads[i] = if (i < pads.len) @as(isize, @intCast(pads[i])) else 0; // begin
                effective_pads[i + spatial_dims] = if (i + spatial_dims < pads.len) @as(isize, @intCast(pads[i + spatial_dims])) else 0; // end
            }
        },
        .VALID => {
            // No padding
            @memset(effective_pads, 0);
        },
        .SAME_UPPER, .SAME_LOWER => {
            // IMPORTANT: Ignore the pads parameter completely when auto_pad is SAME_*
            // Calculate padding using ONNX v22 formula:
            // pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
            for (0..spatial_dims) |i| {
                const input_size = @as(isize, @intCast(input.shape[2 + i]));
                const output_size = @as(isize, @intCast(output.shape[2 + i]));
                const kernel_size = @as(isize, @intCast(kernel_shape[i]));
                const stride = @as(isize, @intCast(strides[i]));
                const dilation = @as(isize, @intCast(dilations[i]));

                // Effective kernel size with dilation: (kernel_shape[i] - 1) * dilations[i] + 1
                const effective_kernel_size = (kernel_size - 1) * dilation + 1;

                // ONNX v22 formula for SAME padding
                const total_pad = @max(0, (output_size - 1) * stride + effective_kernel_size - input_size);

                if (auto_pad == .SAME_UPPER) {
                    // Extra padding at the end for SAME_UPPER
                    effective_pads[i] = total_pad / 2; // begin
                    effective_pads[i + spatial_dims] = total_pad - effective_pads[i]; // end
                } else {
                    // Extra padding at the beginning for SAME_LOWER
                    effective_pads[i + spatial_dims] = total_pad / 2; // end
                    effective_pads[i] = total_pad - effective_pads[i + spatial_dims]; // begin
                }
            }
        },
    }

    // ... rest of your pooling implementation remains the same

    // Calculate strides for efficient indexing
    var input_strides = try pkg_allocator.alloc(usize, input_rank);
    defer pkg_allocator.free(input_strides);
    var output_strides = try pkg_allocator.alloc(usize, output_rank);
    defer pkg_allocator.free(output_strides);

    input_strides[input_rank - 1] = 1;
    output_strides[output_rank - 1] = 1;

    var i: usize = input_rank - 1;
    while (i > 0) {
        i -= 1;
        input_strides[i] = input_strides[i + 1] * input.shape[i + 1];
        output_strides[i] = output_strides[i + 1] * output.shape[i + 1];
    }

    // Main pooling loop - process each batch and channel
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            const base_input_idx = n * input_strides[0] + c * input_strides[1];
            const base_output_idx = n * output_strides[0] + c * output_strides[1];

            // Process all spatial output positions
            try processAveragePoolingSpatial(
                T,
                input,
                output,
                kernel_shape,
                strides,
                dilations,
                effective_pads,
                input_strides[2..],
                output_strides[2..],
                base_input_idx,
                base_output_idx,
                count_include_pad,
            );
        }
    }
}

/// Process spatial dimensions for average pooling using ONNX specification
fn processAveragePoolingSpatial(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const isize,
    input_strides: []const usize,
    output_strides: []const usize,
    base_input_idx: usize,
    base_output_idx: usize,
    count_include_pad: bool,
) !void {
    const spatial_dims = kernel_shape.len;

    // Iterate directly over output coordinates using nested loops
    switch (spatial_dims) {
        1 => {
            for (0..output.shape[2]) |d0| {
                try processPoolWindow(T, input, output, kernel_shape, strides, dilations, pads, input_strides, output_strides, base_input_idx, base_output_idx, count_include_pad, &[_]usize{d0});
            }
        },
        2 => {
            for (0..output.shape[2]) |d0| {
                for (0..output.shape[3]) |d1| {
                    try processPoolWindow(T, input, output, kernel_shape, strides, dilations, pads, input_strides, output_strides, base_input_idx, base_output_idx, count_include_pad, &[_]usize{ d0, d1 });
                }
            }
        },
        3 => {
            for (0..output.shape[2]) |d0| {
                for (0..output.shape[3]) |d1| {
                    for (0..output.shape[4]) |d2| {
                        try processPoolWindow(T, input, output, kernel_shape, strides, dilations, pads, input_strides, output_strides, base_input_idx, base_output_idx, count_include_pad, &[_]usize{ d0, d1, d2 });
                    }
                }
            }
        },
        else => return error.UnsupportedSpatialDims,
    }
}

fn processPoolWindow(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const isize,
    input_strides: []const usize,
    output_strides: []const usize,
    base_input_idx: usize,
    base_output_idx: usize,
    count_include_pad: bool,
    output_coords: []const usize,
) !void {
    const spatial_dims = kernel_shape.len;

    var sum: T = 0;
    var count: usize = 0;

    // Calculate output index
    var output_idx = base_output_idx;
    for (0..spatial_dims) |i| {
        output_idx += output_coords[i] * output_strides[i];
    }

    // Iterate through kernel using nested loops
    switch (spatial_dims) {
        1 => {
            for (0..kernel_shape[0]) |k0| {
                const input_pos_0 = @as(isize, @intCast(output_coords[0])) * @as(isize, @intCast(strides[0])) +
                    @as(isize, @intCast(k0)) * @as(isize, @intCast(dilations[0])) - pads[0];

                if (input_pos_0 >= 0 and input_pos_0 < @as(isize, @intCast(input.shape[2]))) {
                    const input_idx = base_input_idx + @as(usize, @intCast(input_pos_0)) * input_strides[0];
                    if (input_idx < input.data.len) {
                        sum += input.data[input_idx];
                        count += 1;
                    }
                } else if (count_include_pad) {
                    count += 1;
                }
            }
        },
        2 => {
            for (0..kernel_shape[0]) |k0| {
                for (0..kernel_shape[1]) |k1| {
                    const input_pos_0 = @as(isize, @intCast(output_coords[0])) * @as(isize, @intCast(strides[0])) +
                        @as(isize, @intCast(k0)) * @as(isize, @intCast(dilations[0])) - pads[0];
                    const input_pos_1 = @as(isize, @intCast(output_coords[1])) * @as(isize, @intCast(strides[1])) +
                        @as(isize, @intCast(k1)) * @as(isize, @intCast(dilations[1])) - pads[1];

                    if (input_pos_0 >= 0 and input_pos_0 < @as(isize, @intCast(input.shape[2])) and
                        input_pos_1 >= 0 and input_pos_1 < @as(isize, @intCast(input.shape[3])))
                    {
                        const input_idx = base_input_idx +
                            @as(usize, @intCast(input_pos_0)) * input_strides[0] +
                            @as(usize, @intCast(input_pos_1)) * input_strides[1];
                        if (input_idx < input.data.len) {
                            sum += input.data[input_idx];
                            count += 1;
                        }
                    } else if (count_include_pad) {
                        count += 1;
                    }
                }
            }
        },
        3 => {
            for (0..kernel_shape[0]) |k0| {
                for (0..kernel_shape[1]) |k1| {
                    for (0..kernel_shape[2]) |k2| {
                        const input_pos_0 = @as(isize, @intCast(output_coords[0])) * @as(isize, @intCast(strides[0])) +
                            @as(isize, @intCast(k0)) * @as(isize, @intCast(dilations[0])) - pads[0];
                        const input_pos_1 = @as(isize, @intCast(output_coords[1])) * @as(isize, @intCast(strides[1])) +
                            @as(isize, @intCast(k1)) * @as(isize, @intCast(dilations[1])) - pads[1];
                        const input_pos_2 = @as(isize, @intCast(output_coords[2])) * @as(isize, @intCast(strides[2])) +
                            @as(isize, @intCast(k2)) * @as(isize, @intCast(dilations[2])) - pads[2];

                        if (input_pos_0 >= 0 and input_pos_0 < @as(isize, @intCast(input.shape[2])) and
                            input_pos_1 >= 0 and input_pos_1 < @as(isize, @intCast(input.shape[3])) and
                            input_pos_2 >= 0 and input_pos_2 < @as(isize, @intCast(input.shape[4])))
                        {
                            const input_idx = base_input_idx +
                                @as(usize, @intCast(input_pos_0)) * input_strides[0] +
                                @as(usize, @intCast(input_pos_1)) * input_strides[1] +
                                @as(usize, @intCast(input_pos_2)) * input_strides[2];
                            if (input_idx < input.data.len) {
                                sum += input.data[input_idx];
                                count += 1;
                            }
                        } else if (count_include_pad) {
                            count += 1;
                        }
                    }
                }
            }
        },
        else => return error.UnsupportedSpatialDims,
    }

    if (output_idx < output.data.len) {
        output.data[output_idx] = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
    }
}

/// Full ONNX AveragePool with automatic output allocation
/// This function handles ceil_mode during output shape calculation
pub fn onnx_averagepool(
    comptime T: type,
    input: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
    count_include_pad: bool,
) !Tensor(T) {
    // Calculate output shape using ceil_mode
    const output_shape = try get_onnx_averagepool_output_shape(
        input.shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
    defer pkg_allocator.free(output_shape);

    // Allocate output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Call lean version (ceil_mode already used in shape calculation)
    try lean_onnx_averagepool(
        T,
        input,
        &output,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        count_include_pad,
    );

    return output;
}

/// Calculate output shape for ONNX AveragePool operation
/// This is where ceil_mode is actually used
pub fn get_onnx_averagepool_output_shape(
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
                // Explicit padding: ONNX format [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                var pad_begin: usize = 0;
                var pad_end: usize = 0;
                if (i < pads.len) pad_begin = pads[i];
                if (i + spatial_dims < pads.len) pad_end = pads[i + spatial_dims];

                // ONNX formula with dilations:
                // output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
                const effective_kernel_size = dilation * (kernel_size - 1) + 1;
                const total_pad = pad_begin + pad_end;
                const padded_input_size = input_size + total_pad;

                if (padded_input_size >= effective_kernel_size) {
                    const numerator = padded_input_size - effective_kernel_size;
                    if (ceil_mode) {
                        output_size = (numerator + stride) / stride; // Ceiling division
                    } else {
                        output_size = (numerator / stride) + 1;
                    }
                } else {
                    output_size = 1; // Minimum output size
                }
            },
            .VALID => {
                // No padding - ONNX formula for VALID:
                // output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
                const effective_kernel_size = (kernel_size - 1) * dilation + 1;
                if (input_size >= effective_kernel_size) {
                    const numerator = input_size - effective_kernel_size;
                    if (ceil_mode) {
                        output_size = (numerator + stride) / stride + 1; // ceil((input - effective_kernel + 1) / stride)
                    } else {
                        output_size = (numerator / stride) + 1;
                    }
                } else {
                    output_size = 1;
                }
            },
            .SAME_UPPER, .SAME_LOWER => {
                // For SAME padding, output size is calculated first using ONNX formula:
                // SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
                // When ceil_mode is disabled: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1

                if (ceil_mode) {
                    // ceil(input_size / stride) = (input_size + stride - 1) / stride
                    output_size = (input_size + stride - 1) / stride;
                } else {
                    // floor((input_size - 1) / stride) + 1
                    output_size = if (input_size > 0) ((input_size - 1) / stride) + 1 else 1;
                }
            },
        }

        output_shape[2 + i] = output_size;
    }

    return output_shape;
}
