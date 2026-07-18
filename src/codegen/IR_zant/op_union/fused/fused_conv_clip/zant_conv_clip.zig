const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

const conv_lean = IR_zant.core.math_standard.conv_lean;

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
        return error.InvalidDimensions;
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
    if (in_channels % actual_group != 0) return error.InvalidGroupParameter;
    if (out_channels % actual_group != 0) return error.InvalidGroupParameter;
    if (weight_in_channels != in_channels / actual_group) return error.InvalidDimensions;

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
