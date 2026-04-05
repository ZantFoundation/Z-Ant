// Increase evaluation branch quota for complex convolution operations
comptime {
    @setEvalBranchQuota(100000);
}

/// Multidim Conv
/// INPUT:
///     INPUT[input.shape.len - 4] -> batches
///     INPUT[input.shape.len - 3] -> input channels
///     INPUT[input.shape.len - 2] -> rows
///     INPUT[input.shape.len - 1] -> cols
/// KERNEL:
///     KERNEL[kernel.shape.len - 4] -> filters
///     KERNEL[kernel.shape.len - 3] -> channels
///     KERNEL[kernel.shape.len - 2] -> rows
///     KERNEL[kernel.shape.len - 1] -> cols
/// OUTPUT:
///     OUTPUT[output.shape.len - 4] -> input_batch
///     OUTPUT[output.shape.len - 3] -> output channels (number_of_kernel_filters)
///     OUTPUT[output.shape.len - 2] -> rows
///     OUTPUT[output.shape.len - 1] -> cols
/// Convolution tensor with bias
/// TODO: create 2d convolution, atm is 3 or more dimensions
/// TODO: add better check on output size wrt input and kernel
///
///
///
const std = @import("std");
const zant = @import("../../../zant.zig");
const accelerators = @import("../Accelerators/mod.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

const utils_conv = @import("utils_conv.zig");
const calculateOutputShape = utils_conv.get_conv_output_shape;

/// ONNX Conv operation - creates output tensor and performs convolution
/// Following ONNX Conv-22 specification exactly
pub fn conv(
    comptime T: type,
    input: *const Tensor(T), // X: Input tensor [N, C, H, W] or [C, H, W]
    weight: *const Tensor(T), // W: Weight tensor [M, C/group, kH, kW]
    bias: ?*const Tensor(T), // B: Optional bias tensor [M]
    stride: ?[]const usize, // Stride along each spatial axis
    pads: ?[]const usize, // Padding [h_begin, w_begin, h_end, w_end]
    dilations: ?[]const usize, // Dilation along each spatial axis
    group: ?usize, // Number of groups (default 1)
    auto_pad: ?[]const u8, // NOTSET, VALID, SAME_UPPER, SAME_LOWER
) !Tensor(T) {
    // Input validation
    if (input.shape.len != 3 and input.shape.len != 4) {
        return error.InvalidDimensions;
    }
    if (weight.shape.len != 4) {
        return error.InvalidDimensions;
    }

    // Handle 3D input by assuming batch size = 1
    var input_shape: [4]usize = undefined;
    var temp_input: ?Tensor(T) = null;
    var input_ptr = input;

    if (input.shape.len == 3) {
        input_shape[0] = 1; // batch
        input_shape[1] = input.shape[0]; // channels
        input_shape[2] = input.shape[1]; // height
        input_shape[3] = input.shape[2]; // width

        const temp = try Tensor(T).fromArray(&pkg_allocator, input.data, &input_shape);
        temp_input = temp;
        input_ptr = &temp_input.?;
    } else {
        @memcpy(&input_shape, input.shape[0..4]);
    }
    defer if (temp_input) |*t| t.deinit();

    // Calculate output shape
    const output_shape = try calculateOutputShape(T, &input_shape, weight.shape, stride, pads, dilations, auto_pad);

    // Create output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Perform convolution
    try conv_lean(T, input_ptr, weight, &output, bias, stride, pads, dilations, group, auto_pad);

    return output;
}

/// ONNX Conv operation - lean version that writes to pre-allocated output tensor
/// This is the core implementation following ONNX Conv-22 specification
pub fn conv_lean(
    comptime T: type,
    input: *const Tensor(T), // X: Input tensor [N, C, H, W]
    weight: *const Tensor(T), // W: Weight tensor [M, C/group, kH, kW]
    output: *Tensor(T), // Y: Output tensor [N, M, oH, oW]
    bias: ?*const Tensor(T), // B: Optional bias tensor [M]
    stride: ?[]const usize, // Stride along each spatial axis
    pads: ?[]const usize, // Padding [h_begin, w_begin, h_end, w_end]
    dilations: ?[]const usize, // Dilation along each spatial axis
    group: ?usize, // Number of groups (default 1)
    auto_pad: ?[]const u8, // NOTSET, VALID, SAME_UPPER, SAME_LOWER
) !void {
    // Validate input shapes
    if (input.shape.len != 4 or weight.shape.len != 4 or output.shape.len != 4) {
        return error.InvalidDimensions;
    }

    // Extract dimensions
    const batch_size = input.shape[0]; // N
    const in_channels = input.shape[1]; // C
    const in_height = input.shape[2]; // H
    const in_width = input.shape[3]; // W

    const out_channels = weight.shape[0]; // M
    const weight_in_channels = weight.shape[1]; // C/group
    const kernel_height = weight.shape[2]; // kH
    const kernel_width = weight.shape[3]; // kW

    const out_height = output.shape[2]; // oH
    const out_width = output.shape[3]; // oW

    // Validate and set defaults
    const actual_group = group orelse 1;
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Group validation
    if (in_channels % actual_group != 0) {
        return error.InvalidGroupParameter;
    }
    if (out_channels % actual_group != 0) {
        return error.InvalidGroupParameter;
    }
    if (weight_in_channels != in_channels / actual_group) {
        return error.InvalidDimensions;
    }

    const channels_per_group = in_channels / actual_group;
    const filters_per_group = out_channels / actual_group;

    // Calculate padding
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            // No padding
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            // Calculate padding for SAME
            const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
            const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

            const total_pad_h = if (out_height * stride_h + dilated_kernel_h > in_height)
                (out_height - 1) * stride_h + dilated_kernel_h - in_height
            else
                0;
            const total_pad_w = if (out_width * stride_w + dilated_kernel_w > in_width)
                (out_width - 1) * stride_w + dilated_kernel_w - in_width
            else
                0;

            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = total_pad_h / 2;
                pad_h_end = total_pad_h - pad_h_begin;
                pad_w_begin = total_pad_w / 2;
                pad_w_end = total_pad_w - pad_w_begin;
            } else { // SAME_LOWER
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;
                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        } else if (!std.mem.eql(u8, pad_mode, "NOTSET")) {
            return error.InvalidPadding;
        }
    }

    // Use explicit padding if provided and auto_pad is NOTSET or null
    if (pads) |p| {
        if (p.len >= 4) {
            pad_h_begin = p[0];
            pad_w_begin = p[1];
            pad_h_end = p[2];
            pad_w_end = p[3];
        } else if (p.len == 2) {
            pad_h_begin = p[0];
            pad_h_end = p[0];
            pad_w_begin = p[1];
            pad_w_end = p[1];
        }
    }

    const auto_pad_mode: accelerators.AutoPadMode = blk: {
        if (auto_pad) |pad_mode| {
            if (std.mem.eql(u8, pad_mode, "VALID")) break :blk .valid;
            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) break :blk .same_upper;
            if (std.mem.eql(u8, pad_mode, "SAME_LOWER")) break :blk .same_lower;
        }
        break :blk .notset;
    };

    // Bias array for efficient access
    var bias_data: ?[]const T = null;
    if (bias) |b| {
        if (b.shape.len != 1 or b.shape[0] != out_channels) {
            return error.InvalidDimensions;
        }
        bias_data = b.data;
    }

    const conv_params = accelerators.ConvPreparedParams{
        .stride = .{ stride_h, stride_w },
        .dilations = .{ dilation_h, dilation_w },
        .pads = .{ pad_h_begin, pad_w_begin, pad_h_end, pad_w_end },
        .group = actual_group,
        .filters_per_group = filters_per_group,
        .channels_per_group = channels_per_group,
        .auto_pad = auto_pad_mode,
    };

    if (try accelerators.tryConvLean(T, input, weight, output, bias_data, conv_params)) {
        return;
    }

    // Initialize output to zero
    try output.set(0, 0);

    // Main convolution loop
    // Process each batch
    for (0..batch_size) |n| {
        // Process each output channel
        for (0..out_channels) |m| {
            const current_group = m / filters_per_group;
            const in_channel_start = current_group * channels_per_group;
            const in_channel_end = in_channel_start + channels_per_group;

            // Get bias value for this output channel
            const bias_val: T = if (bias_data) |b| b[m] else 0;

            // Process each output spatial location
            for (0..out_height) |oh| {
                for (0..out_width) |ow| {
                    var sum: T = bias_val;

                    // Calculate input region for this output location
                    const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));
                    const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));

                    // Convolution over kernel
                    for (0..kernel_height) |kh| {
                        for (0..kernel_width) |kw| {
                            const in_h = in_h_start + @as(isize, @intCast(kh * dilation_h));
                            const in_w = in_w_start + @as(isize, @intCast(kw * dilation_w));

                            // Check bounds
                            if (in_h >= 0 and in_h < @as(isize, @intCast(in_height)) and
                                in_w >= 0 and in_w < @as(isize, @intCast(in_width)))
                            {
                                const ih = @as(usize, @intCast(in_h));
                                const iw = @as(usize, @intCast(in_w));

                                // Sum over input channels in this group
                                for (in_channel_start..in_channel_end) |c| {
                                    const k_c = c - in_channel_start; // Map to weight channel index

                                    const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                                    const weight_idx = ((m * weight_in_channels + k_c) * kernel_height + kh) * kernel_width + kw;

                                    sum += input.data[input_idx] * weight.data[weight_idx];
                                }
                            }
                            // Note: for padded regions (out of bounds), we implicitly add 0
                        }
                    }

                    // Store result
                    const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                    output.data[output_idx] = sum;
                }
            }
        }
    }
}
