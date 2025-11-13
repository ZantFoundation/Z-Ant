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
pub var log_functionC: ?*const fn ([*c]u8) callconv(.c) void = null;

pub export fn setLogFunctionC(func: ?*const fn ([*c]u8) callconv(.c) void) void {
    log_functionC = func;
}

const std = @import("std");
const zant = @import("../../../zant.zig");
const accelerators = @import("../Accelerators/mod.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

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

                    // Store result
                    const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                    output.data[output_idx] = sum;
                }
            }
        }
    }
}

pub fn calculateOutputShape(
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

// Test functions for validation
test "conv basic functionality" {
    const testing = std.testing;

    // Create simple test tensors
    var input_shape = [_]usize{ 1, 1, 3, 3 }; // [N=1, C=1, H=3, W=3]
    var input = try Tensor(f32).fromShape(&pkg_allocator, &input_shape);
    defer input.deinit();

    // Fill input with simple values
    for (0..9) |i| {
        input.data[i] = @as(f32, @floatFromInt(i + 1));
    }

    var weight_shape = [_]usize{ 1, 1, 2, 2 }; // [M=1, C=1, kH=2, kW=2]
    var weight = try Tensor(f32).fromShape(&pkg_allocator, &weight_shape);
    defer weight.deinit();

    // Simple 2x2 kernel
    weight.data[0] = 1.0;
    weight.data[1] = 0.0;
    weight.data[2] = 0.0;
    weight.data[3] = 1.0;

    // Test convolution
    var result = try conv(f32, &input, &weight, null, null, null, null, null, null);
    defer result.deinit();

    // Verify output shape [1, 1, 2, 2]
    try testing.expect(result.shape.len == 4);
    try testing.expect(result.shape[0] == 1);
    try testing.expect(result.shape[1] == 1);
    try testing.expect(result.shape[2] == 2);
    try testing.expect(result.shape[3] == 2);

    // Verify some values (diagonal kernel should add diagonal elements)
    try testing.expect(result.data[0] == 6.0); // 1+5
    try testing.expect(result.data[1] == 8.0); // 2+6
    try testing.expect(result.data[2] == 12.0); // 4+8
    try testing.expect(result.data[3] == 14.0); // 5+9
}

test "conv with stride and padding" {
    const testing = std.testing;

    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input = try Tensor(f32).fromShape(&pkg_allocator, &input_shape);
    defer input.deinit();

    for (0..16) |i| {
        input.data[i] = 1.0;
    }

    var weight_shape = [_]usize{ 1, 1, 3, 3 };
    var weight = try Tensor(f32).fromShape(&pkg_allocator, &weight_shape);
    defer weight.deinit();

    for (0..9) |i| {
        weight.data[i] = 1.0;
    }

    const stride_vals = [_]usize{ 2, 2 };
    const pad_vals = [_]usize{ 1, 1, 1, 1 };

    var result = try conv(f32, &input, &weight, null, &stride_vals, &pad_vals, null, null, null);
    defer result.deinit();

    // With stride=2 and padding=1, output should be [1, 1, 2, 2]
    try testing.expect(result.shape[2] == 2);
    try testing.expect(result.shape[3] == 2);
}

/// TRUE Conv+Clip FUSION - modifies the core conv loop to clip inline
/// NO separate passes = maximum cache efficiency
/// PERFORMANCE CRITICAL: Force aggressive optimization
pub fn conv_clip_lean(
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
    min_tensor: ?*const Tensor(T), // Min clipping value (typically 0 for ReLU)
    max_tensor: ?*const Tensor(T), // Max clipping value (typically 6 for ReLU6)
) !void {
    // Extract clip values
    var clip_min: T = std.math.floatMin(T);
    var clip_max: T = std.math.floatMax(T);

    if (min_tensor) |min_t| {
        if (min_t.data.len > 0) clip_min = min_t.data[0];
    }

    if (max_tensor) |max_t| {
        if (max_t.data.len > 0) clip_max = max_t.data[0];
    }

    // If no clipping, just call regular conv
    if (clip_min <= std.math.floatMin(T) and clip_max >= std.math.floatMax(T)) {
        return conv_lean(T, input, weight, output, bias, stride, pads, dilations, group, auto_pad);
    }

    // SIMPLE OPTIMIZATION: Skip unnecessary checks for common case

    // INLINE FUSION: Copy conv_lean code but modify the output store to clip inline
    // This avoids the extra memory pass completely

    // Validate input shapes
    if (input.shape.len != 4 or weight.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    // Extract dimensions (same as conv_lean)
    const batch_size = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];

    const out_channels = weight.shape[0];
    const weight_in_channels = weight.shape[1];
    const kernel_height = weight.shape[2];
    const kernel_width = weight.shape[3];

    const out_height = output.shape[2];
    const out_width = output.shape[3];

    // Validate and set defaults (same as conv_lean)
    const actual_group = group orelse 1;
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Group validation (same as conv_lean)
    if (in_channels % actual_group != 0) return TensorMathError.InvalidGroupParameter;
    if (out_channels % actual_group != 0) return TensorMathError.InvalidGroupParameter;
    if (weight_in_channels != in_channels / actual_group) return TensorMathError.InvalidDimensions;

    const channels_per_group = in_channels / actual_group;
    const filters_per_group = out_channels / actual_group;

    // Calculate padding (copy from conv_lean - essential for correctness)
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
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
            } else {
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;
                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        }
    }

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

    // Winograd disabled - too much overhead for MobileNet sizes

    // OPTIMIZED FUSED LOOP - baseline version that achieved 1.61s
    for (0..batch_size) |n| {
        for (0..out_channels) |m| {
            const group_idx = m / filters_per_group;
            const in_channel_start = group_idx * channels_per_group;
            const in_channel_end = in_channel_start + channels_per_group;

            // Pre-calculate bias once per channel
            const bias_val: T = if (bias) |b| b.data[m] else 0;

            for (0..out_height) |oh| {
                // Pre-calculate input height base for this output row
                const ih_base = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                for (0..out_width) |ow| {
                    var sum: T = bias_val; // Start with bias

                    // Pre-calculate input width base for this output pixel
                    const iw_base = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));

                    // Optimized kernel loops with reduced calculations
                    for (0..kernel_height) |kh| {
                        const ih = ih_base + @as(isize, @intCast(kh * dilation_h));

                        // Early exit if outside bounds
                        if (ih < 0 or ih >= @as(isize, @intCast(in_height))) continue;

                        const ih_usize = @as(usize, @intCast(ih));

                        for (0..kernel_width) |kw| {
                            const iw = iw_base + @as(isize, @intCast(kw * dilation_w));

                            // Early exit if outside bounds
                            if (iw < 0 or iw >= @as(isize, @intCast(in_width))) continue;

                            const iw_usize = @as(usize, @intCast(iw));

                            // Inner channel loop - this is where most time is spent
                            for (in_channel_start..in_channel_end) |c| {
                                const k_c = c - in_channel_start;

                                // Optimized index calculations
                                const input_idx = ((n * in_channels + c) * in_height + ih_usize) * in_width + iw_usize;
                                const weight_idx = ((m * weight_in_channels + k_c) * kernel_height + kh) * kernel_width + kw;

                                // Accumulate - compiler auto-vectorizes this well
                                sum += input.data[input_idx] * weight.data[weight_idx];
                            }
                        }
                    }

                    // FUSION POINT: Store with inline clipping
                    const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                    output.data[output_idx] = @min(clip_max, @max(clip_min, sum));
                }
            }
        }
    }
}

/// WINOGRAD F(2,3) Convolution with Clipping
/// Transforms 3x3 kernel to reduce multiplications from 9 to 4 per output pixel
/// Input: 4x4 tile → Output: 2x2 tile, Kernel: 3x3 → Transform: 4x4
inline fn conv3x3_winograd_clip_lean(
    comptime T: type,
    input: *const Tensor(T),
    weight: *const Tensor(T),
    output: *Tensor(T),
    bias: ?*const Tensor(T),
    pad_h: usize,
    pad_w: usize,
    clip_min: T,
    clip_max: T,
    actual_group: usize,
    channels_per_group: usize,
    filters_per_group: usize,
) !void {
    // Parameters used in channel grouping calculations below
    const batch_size = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = weight.shape[0];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    // Winograd F(2,3) transformation matrices
    // G: transforms 3x3 kernel to 4x4
    const G = [4][3]T{
        [3]T{ 1.0, 0.0, 0.0 },
        [3]T{ 0.5, 0.5, 0.5 },
        [3]T{ 0.5, -0.5, 0.5 },
        [3]T{ 0.0, 0.0, 1.0 },
    };

    // BT: transforms 4x4 input tile
    const BT = [4][4]T{
        [4]T{ 1.0, 0.0, -1.0, 0.0 },
        [4]T{ 0.0, 1.0, 1.0, 0.0 },
        [4]T{ 0.0, -1.0, 1.0, 0.0 },
        [4]T{ 0.0, 1.0, 0.0, -1.0 },
    };

    // AT: transforms 4x4 result to 2x2 output
    const AT = [2][4]T{
        [4]T{ 1.0, 1.0, 1.0, 0.0 },
        [4]T{ 0.0, 1.0, -1.0, -1.0 },
    };

    // Process in 2x2 output tiles
    const tile_h = 2;
    const tile_w = 2;
    const n_tiles_h = (out_height + tile_h - 1) / tile_h;
    const n_tiles_w = (out_width + tile_w - 1) / tile_w;

    for (0..batch_size) |n| {
        for (0..out_channels) |m| {
            const group_idx = m / filters_per_group;
            const in_channel_start = group_idx * channels_per_group;
            const in_channel_end = in_channel_start + channels_per_group;

            // Validate group parameters are used correctly
            if (group_idx >= actual_group) continue;

            const bias_val: T = if (bias) |b| b.data[m] else 0;

            // Transform weight once per output channel
            var transformed_weight: [4][4]T = undefined;
            for (0..4) |i| {
                for (0..4) |j| {
                    transformed_weight[i][j] = 0;
                    for (0..3) |k| {
                        for (0..3) |l| {
                            // Get weight value for this channel group
                            for (in_channel_start..in_channel_end) |c| {
                                const k_c = c - in_channel_start;
                                const weight_idx = ((m * channels_per_group + k_c) * 3 + k) * 3 + l;
                                transformed_weight[i][j] += G[i][k] * weight.data[weight_idx] * G[j][l];
                            }
                        }
                    }
                }
            }

            // Process each 2x2 output tile
            for (0..n_tiles_h) |tile_y| {
                for (0..n_tiles_w) |tile_x| {
                    // Extract 4x4 input tile (with padding)
                    var input_tile: [4][4]T = [_][4]T{[_]T{0} ** 4} ** 4;

                    const base_y = @as(isize, @intCast(tile_y * tile_h)) - @as(isize, @intCast(pad_h));
                    const base_x = @as(isize, @intCast(tile_x * tile_w)) - @as(isize, @intCast(pad_w));

                    for (0..4) |i| {
                        for (0..4) |j| {
                            const y = base_y + @as(isize, @intCast(i));
                            const x = base_x + @as(isize, @intCast(j));

                            if (y >= 0 and y < @as(isize, @intCast(in_height)) and
                                x >= 0 and x < @as(isize, @intCast(in_width)))
                            {
                                // Sum across input channels for this group
                                for (in_channel_start..in_channel_end) |c| {
                                    const input_idx = ((n * in_channels + c) * in_height + @as(usize, @intCast(y))) * in_width + @as(usize, @intCast(x));
                                    input_tile[i][j] += input.data[input_idx];
                                }
                            }
                        }
                    }

                    // Transform input tile: BT * input_tile * B
                    var transformed_input: [4][4]T = undefined;
                    for (0..4) |i| {
                        for (0..4) |j| {
                            transformed_input[i][j] = 0;
                            for (0..4) |k| {
                                for (0..4) |l| {
                                    transformed_input[i][j] += BT[i][k] * input_tile[k][l] * BT[j][l];
                                }
                            }
                        }
                    }

                    // Element-wise multiplication (the core Winograd computation)
                    var multiplied: [4][4]T = undefined;
                    for (0..4) |i| {
                        for (0..4) |j| {
                            multiplied[i][j] = transformed_input[i][j] * transformed_weight[i][j];
                        }
                    }

                    // Inverse transform: AT * multiplied * A to get 2x2 output
                    var output_tile: [2][2]T = undefined;
                    for (0..2) |i| {
                        for (0..2) |j| {
                            output_tile[i][j] = 0;
                            for (0..4) |k| {
                                for (0..4) |l| {
                                    output_tile[i][j] += AT[i][k] * multiplied[k][l] * AT[j][l];
                                }
                            }

                            // Add bias and apply clipping
                            output_tile[i][j] += bias_val;
                            output_tile[i][j] = @min(clip_max, @max(clip_min, output_tile[i][j]));
                        }
                    }

                    // Store 2x2 output tile back to output tensor
                    for (0..2) |i| {
                        for (0..2) |j| {
                            const out_y = tile_y * tile_h + i;
                            const out_x = tile_x * tile_w + j;

                            if (out_y < out_height and out_x < out_width) {
                                const output_idx = ((n * out_channels + m) * out_height + out_y) * out_width + out_x;
                                output.data[output_idx] = output_tile[i][j];
                            }
                        }
                    }
                }
            }
        }
    }
}
