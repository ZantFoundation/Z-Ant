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
const std = @import("std");
const zant = @import("../../../zant.zig");
const accelerators = @import("../Accelerators/mod.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

/// ONNX Conv+ReLU operation - creates output tensor and performs convolution
pub fn conv_relu(
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
        return TensorMathError.InvalidDimensions;
    }
    if (weight.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
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
    const output_shape = try get_conv_relu_output_shape(T, pkg_allocator, &input_shape, weight.shape, stride, pads, dilations, auto_pad);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output_tensor.deinit();

    // Perform convolution
    try conv_relu_lean(T, input_ptr, weight, &output_tensor, bias, stride, pads, dilations, group, auto_pad);

    return output_tensor;
}

/// ONNX Conv+ReLU operation - lean version that writes to pre-allocated output tensor
pub fn conv_relu_lean(
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
        return TensorMathError.InvalidDimensions;
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
        return TensorMathError.InvalidGroupParameter;
    }
    if (out_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (weight_in_channels != in_channels / actual_group) {
        return TensorMathError.InvalidDimensions;
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
            return TensorMathError.InvalidPadding;
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
            return TensorMathError.InvalidDimensions;
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

                    // Compute the index for storing the result
                    const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;

                    // Store result applying ReLU
                    output.data[output_idx] = @max(0, sum);
                }
            }
        }
    }
}

// Compute output shape
pub inline fn get_conv_relu_output_shape(
    comptime T: type,
    allocator: std.mem.Allocator,
    input_shape: []const usize, // [N, C, H, W]
    weight_shape: []const usize, // [M, C/group, kH, kW]
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    // group: ?usize,
    auto_pad: ?[]const u8,
) ![]usize {
    _ = T; // Suppress unused parameter warning
    if (input_shape.len != 4 or weight_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }
    const batch_size = input_shape[0];
    const in_height = input_shape[2];
    const in_width = input_shape[3];
    const out_channels = weight_shape[0];
    const kernel_height = weight_shape[2];
    const kernel_width = weight_shape[3];

    // Set defaults
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Calculate effective kernel size with dilation
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            // No padding - already initialized to 0
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            // For SAME padding, output size should be ceil(input_size / stride)
            const out_height = (in_height + stride_h - 1) / stride_h;
            const out_width = (in_width + stride_w - 1) / stride_w;
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
            return TensorMathError.InvalidPadding;
        }
    }

    // Use explicit padding if provided
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

    // Calculate output dimensions
    const padded_height = in_height + pad_h_begin + pad_h_end;
    const padded_width = in_width + pad_w_begin + pad_w_end;

    if (padded_height < dilated_kernel_h or padded_width < dilated_kernel_w) {
        return TensorMathError.InvalidDimensions;
    }

    const out_height = (padded_height - dilated_kernel_h) / stride_h + 1;
    const out_width = (padded_width - dilated_kernel_w) / stride_w + 1;

    // Allocate and return slice
    const output_shape = try allocator.alloc(usize, 4);
    output_shape[0] = batch_size;
    output_shape[1] = out_channels;
    output_shape[2] = out_height;
    output_shape[3] = out_width;

    return output_shape;
}
