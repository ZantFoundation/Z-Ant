const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("../op_conv/zant_conv.zig");

const utils = @import("utils_qlinearconv.zig");

// Lightweight logger that forwards to the global core tensor log_function (if set)
inline fn coreLogStatic(comptime msg: []const u8) void {
    if (zant.core.tensor.log_function) |log| {
        log(@ptrCast(@constCast(msg)));
    }
}

inline fn coreLogf(comptime fmt: []const u8, args: anytype) void {
    if (zant.core.tensor.log_function) |log| {
        var buf: [256]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
        const n = s.len;
        if (n < buf.len) {
            buf[n] = 0;
        } else {
            buf[buf.len - 1] = 0;
        }
        log(@ptrCast(&buf[0]));
    }
}

/// QLinearConv operation following ONNX specification
/// Performs quantized convolution using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - w: quantized weight tensor
/// - w_scale: scale factor for weight quantization
/// - w_zero_point: zero point for weight quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
/// - bias: optional bias tensor (can be null)
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(conv(dequantize(x), dequantize(w)) + bias, y_scale, y_zero_point)
pub fn qlinear_conv(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    bias: ?*const Tensor(BiasType),
    // Convolution parameters
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: ?[]const u8,
) !Tensor(InputType) {
    // Input validation
    if (x.shape.len != 3 and x.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }
    if (w.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    // Handle 3D input by assuming batch size = 1
    var input_shape: [4]usize = undefined;
    var temp_input: ?Tensor(InputType) = null;
    var input_ptr = x;

    if (x.shape.len == 3) {
        input_shape[0] = 1; // batch
        input_shape[1] = x.shape[0]; // channels
        input_shape[2] = x.shape[1]; // height
        input_shape[3] = x.shape[2]; // width

        const temp = try Tensor(InputType).fromArray(&pkg_allocator, x.data, &input_shape);
        temp_input = temp;
        input_ptr = &temp_input.?;
    } else {
        @memcpy(&input_shape, x.shape[0..4]);
    }
    defer if (temp_input) |*t| t.deinit();

    // Calculate output shape using existing conv calculation
    const output_shape = try conv.calculateOutputShape(InputType, &input_shape, w.shape, stride, pads, dilations, auto_pad);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Perform quantized convolution using dispatch to get the best implementation
    try utils.qlinearconv_dispatch(InputType, WeightType, ScaleType, InputType, BiasType, input_ptr, x_scale, x_zero_point, w, w_scale, w_zero_point, &output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad.?);

    return output;
}

/// OPTIMIZED: Lean version of QLinearConv with pre-computation and cache optimizations
pub fn qlinear_conv_lean(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype, // ZeroPointType unused due to anytype zero_point parameters
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype, // Accept any tensor type for zero_point
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype, // Accept any tensor type for zero_point
    output: *Tensor(InputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype, // Accept any tensor type for zero_point
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    coreLogStatic("QLINEAR: using qlinearconv_lean (fp)\n");
    // DEBUG: Print which function is being called
    // std.debug.print("QLINEAR_DEBUG: using qlinearconv_lean (floating point)\n", .{});
    _ = auto_pad; // not handled: use explicit pads

    // Check tensor shapes
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        // std.log.err("QLinearConv: InvalidDimensions x.shape={any} w.shape={any} y.shape={any}", .{ x.shape, w.shape, output.shape });
        return TensorMathError.InvalidDimensions;
    }

    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    // Handle scalar tensors (shape [1])
    if (x.shape.len == 1 and x.shape[0] == 1 and output.shape.len == 1 and output.shape[0] == 1) {
        output.data[0] = if (isInt(InputType)) @as(InputType, 0) else @as(InputType, 0.0);
        return;
    }

    // Extract dimensions
    const batch_size = x.shape[0]; // N
    const in_channels = x.shape[1]; // C
    const in_height = x.shape[2]; // H
    const in_width = x.shape[3]; // W

    const out_channels = w.shape[0]; // M
    const weight_in_channels = w.shape[1]; // C/group
    const kernel_height = w.shape[2]; // kH
    const kernel_width = w.shape[3]; // kW

    const out_height = output.shape[2]; // oH
    const out_width = output.shape[3]; // oW

    // Parameters
    const actual_group = group orelse 1;
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Groups validation
    if (in_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (out_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (weight_in_channels != in_channels / actual_group) {
        return TensorMathError.InvalidDimensions;
    }

    // Padding
    var pad_h_begin: usize = 0;
    var pad_w_begin: usize = 0;
    if (pads) |p| {
        if (p.len >= 2) {
            pad_h_begin = p[0];
            pad_w_begin = p[1];
        }
    }

    // ===== OPTIMIZATION 1: Pre-compute scale and bias =====
    const x_scale_val: f32 = asF32(ScaleType, x_scale.data[0]);
    const y_scale_val: f32 = asF32(ScaleType, y_scale.data[0]);
    const x_zp_f: f32 = if (x_zero_point.data.len > 0) asF32(@TypeOf(x_zero_point.data[0]), x_zero_point.data[0]) else 0.0;
    const y_zp_f: f32 = if (y_zero_point.data.len > 0) asF32(@TypeOf(y_zero_point.data[0]), y_zero_point.data[0]) else 0.0;

    // Pre-compute scale and bias for all output channels (avoids redundant computation)
    var channel_scales: std.ArrayList(f32) = .empty;
    defer channel_scales.deinit(pkg_allocator);
    var channel_zps: std.ArrayList(f32) = .empty;
    defer channel_zps.deinit(pkg_allocator);
    var channel_bias: std.ArrayList(f32) = .empty;
    defer channel_bias.deinit(pkg_allocator);

    try channel_scales.ensureTotalCapacity(pkg_allocator, out_channels);
    try channel_zps.ensureTotalCapacity(pkg_allocator, out_channels);
    try channel_bias.ensureTotalCapacity(pkg_allocator, out_channels);

    for (0..out_channels) |m| {
        const w_scale_val: f32 = if (w_scale.data.len == out_channels)
            asF32(ScaleType, w_scale.data[m])
        else
            asF32(ScaleType, w_scale.data[0]);

        const w_zp_f: f32 = if (w_zero_point.data.len == out_channels)
            asF32(@TypeOf(w_zero_point.data[0]), w_zero_point.data[m])
        else if (w_zero_point.data.len > 0)
            asF32(@TypeOf(w_zero_point.data[0]), w_zero_point.data[0])
        else
            0.0;

        const bias_f: f32 = if (bias) |b| blk: {
            const b_raw = if (b.data.len == 1) b.data[0] else b.data[m];
            const b_val: f32 = if (isInt(BiasType))
                asF32(BiasType, b_raw) * x_scale_val * w_scale_val
            else
                asF32(BiasType, b_raw);
            break :blk b_val;
        } else 0.0;

        channel_scales.appendAssumeCapacity(w_scale_val);
        channel_zps.appendAssumeCapacity(w_zp_f);
        channel_bias.appendAssumeCapacity(bias_f);
    }

    // Pre-compute quantization bounds
    const q_min: f32 = asF32(InputType, std.math.minInt(InputType));
    const q_max: f32 = asF32(InputType, std.math.maxInt(InputType));

    // ===== OPTIMIZATION 2: Specialized paths for common kernels =====
    if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
        // Optimized for 3x3 (MobileNet style)
        try conv3x3Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else if (kernel_height == 1 and kernel_width == 1) {
        // Optimized for 1x1 (pointwise)
        try conv1x1Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else {
        // ===== OPTIMIZATION 3: Original loop with pre-computations =====
        for (0..batch_size) |n| {
            for (0..actual_group) |g| {
                const in_c_start = g * (in_channels / actual_group);
                const in_c_end = (g + 1) * (in_channels / actual_group);
                const out_c_start = g * (out_channels / actual_group);
                const out_c_end = (g + 1) * (out_channels / actual_group);

                // Process output channels in blocks for better cache locality
                const block_size = 4;
                var m_block = out_c_start;
                while (m_block < out_c_end) {
                    const m_end = @min(m_block + block_size, out_c_end);

                    for (m_block..m_end) |m| {
                        const w_scale_val = channel_scales.items[m];
                        const w_zp_f = channel_zps.items[m];
                        const bias_f = channel_bias.items[m];

                        for (0..out_height) |oh| {
                            const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                            for (0..out_width) |ow| {
                                const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));
                                var acc: f32 = bias_f;

                                for (0..kernel_height) |kh| {
                                    const in_h = in_h_start + @as(isize, @intCast(kh * dilation_h));
                                    if (in_h < 0 or in_h >= @as(isize, @intCast(in_height))) continue;

                                    for (0..kernel_width) |kw| {
                                        const in_w = in_w_start + @as(isize, @intCast(kw * dilation_w));
                                        if (in_w < 0 or in_w >= @as(isize, @intCast(in_width))) continue;

                                        const ih = @as(usize, @intCast(in_h));
                                        const iw = @as(usize, @intCast(in_w));

                                        for (in_c_start..in_c_end) |c| {
                                            const k_c = c - in_c_start;
                                            const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                                            const weight_idx = ((m * weight_in_channels + k_c) * kernel_height + kh) * kernel_width + kw;

                                            const x_real: f32 = if (isInt(InputType)) blk: {
                                                const qx = asF32(InputType, x.data[input_idx]);
                                                break :blk x_scale_val * (qx - x_zp_f);
                                            } else asF32(InputType, x.data[input_idx]);

                                            const w_real: f32 = if (isInt(WeightType)) blk: {
                                                const qw = asF32(WeightType, w.data[weight_idx]);
                                                break :blk w_scale_val * (qw - w_zp_f);
                                            } else asF32(WeightType, w.data[weight_idx]);

                                            acc += x_real * w_real;
                                        }
                                    }
                                }

                                const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                                const q_unrounded: f32 = acc / y_scale_val + y_zp_f;

                                // Use ONNX Runtime compatible rounding: round half away from zero
                                var q_rounded: f32 = undefined;
                                if (q_unrounded >= 0) {
                                    q_rounded = @floor(q_unrounded + 0.5);
                                } else {
                                    q_rounded = @ceil(q_unrounded - 0.5);
                                }

                                const q_clamped = std.math.clamp(q_rounded, q_min, q_max);

                                // DEBUG: Log specific case where we get error (output index 22 which should be 188)
                                // if (output_idx == 22) {
                                //     std.debug.print("DEBUG_FLOAT: idx={} acc={:.10} q_unrounded={:.10} q_rounded={:.10} q_clamped={:.10} final={}\n", .{ output_idx, acc, q_unrounded, q_rounded, q_clamped, @as(InputType, @intFromFloat(q_clamped)) });
                                // }

                                output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                            }
                        }
                    }
                    m_block += block_size;
                }
            }
        }
    }
}

// ===== OPTIMIZED SPECIALIZATIONS =====

/// Optimized 3x3 convolution with loop unrolling and cache blocking
fn conv3x3Optimized(x: anytype, w: anytype, output: anytype, batch_size: usize, actual_group: usize, in_channels: usize, out_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, stride_h: usize, stride_w: usize, pad_h_begin: usize, pad_w_begin: usize, x_scale_val: f32, x_zp_f: f32, channel_scales: []const f32, channel_zps: []const f32, channel_bias: []const f32, y_scale_val: f32, y_zp_f: f32, q_min: f32, q_max: f32, comptime InputType: type, comptime WeightType: type) !void {
    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                const w_scale_val = channel_scales[m];
                const w_zp_f = channel_zps[m];
                const bias_f = channel_bias[m];

                for (0..out_height) |oh| {
                    const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                    for (0..out_width) |ow| {
                        const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));
                        var acc: f32 = bias_f;

                        // Unroll 3x3 kernel manually for better performance
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;

                            // kh=0, kw=0
                            const in_h_0 = in_h_start;
                            const in_w_0 = in_w_start;
                            if (in_h_0 >= 0 and in_h_0 < in_height and in_w_0 >= 0 and in_w_0 < in_width) {
                                const ih_0 = @as(usize, @intCast(in_h_0));
                                const iw_0 = @as(usize, @intCast(in_w_0));
                                const input_idx = ((n * in_channels + c) * in_height + ih_0) * in_width + iw_0;
                                const weight_idx = ((m * weight_in_channels + k_c) * 3 + 0) * 3 + 0;

                                const x_real: f32 = if (isInt(InputType)) blk: {
                                    const qx = asF32(InputType, x.data[input_idx]);
                                    break :blk x_scale_val * (qx - x_zp_f);
                                } else asF32(InputType, x.data[input_idx]);

                                const w_real: f32 = if (isInt(WeightType)) blk: {
                                    const qw = asF32(WeightType, w.data[weight_idx]);
                                    break :blk w_scale_val * (qw - w_zp_f);
                                } else asF32(WeightType, w.data[weight_idx]);

                                acc += x_real * w_real;
                            }

                            // Continue unrolling for all 9 positions (kh=0,1,2 x kw=0,1,2)
                            // Remaining unrolled positions for brevity...
                            inline for (0..3) |kh| {
                                inline for (0..3) |kw| {
                                    if (kh == 0 and kw == 0) continue; // already done above

                                    const in_h = in_h_start + @as(isize, @intCast(kh));
                                    const in_w = in_w_start + @as(isize, @intCast(kw));

                                    if (in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width) {
                                        const ih = @as(usize, @intCast(in_h));
                                        const iw = @as(usize, @intCast(in_w));
                                        const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                                        const weight_idx = ((m * weight_in_channels + k_c) * 3 + kh) * 3 + kw;

                                        const x_real: f32 = if (isInt(InputType)) blk: {
                                            const qx = asF32(InputType, x.data[input_idx]);
                                            break :blk x_scale_val * (qx - x_zp_f);
                                        } else asF32(InputType, x.data[input_idx]);

                                        const w_real: f32 = if (isInt(WeightType)) blk: {
                                            const qw = asF32(WeightType, w.data[weight_idx]);
                                            break :blk w_scale_val * (qw - w_zp_f);
                                        } else asF32(WeightType, w.data[weight_idx]);

                                        acc += x_real * w_real;
                                    }
                                }
                            }
                        }

                        const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                        const q_unrounded: f32 = acc / y_scale_val + y_zp_f;

                        // Use ONNX Runtime compatible rounding: round half away from zero
                        var q_rounded: f32 = undefined;
                        if (q_unrounded >= 0) {
                            q_rounded = @floor(q_unrounded + 0.5);
                        } else {
                            q_rounded = @ceil(q_unrounded - 0.5);
                        }

                        const q_clamped = std.math.clamp(q_rounded, q_min, q_max);
                        output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                    }
                }
            }
        }
    }
}

/// Optimized 1x1 convolution (pointwise - essentially matrix multiplication)
fn conv1x1Optimized(x: anytype, w: anytype, output: anytype, batch_size: usize, actual_group: usize, in_channels: usize, out_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, x_scale_val: f32, x_zp_f: f32, channel_scales: []const f32, channel_zps: []const f32, channel_bias: []const f32, y_scale_val: f32, y_zp_f: f32, q_min: f32, q_max: f32, comptime InputType: type, comptime WeightType: type) !void {
    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                const w_scale_val = channel_scales[m];
                const w_zp_f = channel_zps[m];
                const bias_f = channel_bias[m];

                // 1x1 conv is matrix multiplication - optimized accordingly
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var acc: f32 = bias_f;

                        // No spatial kernel, only channel mixing
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;
                            const input_idx = ((n * in_channels + c) * in_height + oh) * in_width + ow;
                            const weight_idx = m * weight_in_channels + k_c;

                            const x_real: f32 = if (isInt(InputType)) blk: {
                                const qx = asF32(InputType, x.data[input_idx]);
                                break :blk x_scale_val * (qx - x_zp_f);
                            } else asF32(InputType, x.data[input_idx]);

                            const w_real: f32 = if (isInt(WeightType)) blk: {
                                const qw = asF32(WeightType, w.data[weight_idx]);
                                break :blk w_scale_val * (qw - w_zp_f);
                            } else asF32(WeightType, w.data[weight_idx]);

                            acc += x_real * w_real;
                        }

                        const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                        const q_unrounded: f32 = acc / y_scale_val + y_zp_f;

                        // Use ONNX Runtime compatible rounding: round half away from zero
                        var q_rounded: f32 = undefined;
                        if (q_unrounded >= 0) {
                            q_rounded = @floor(q_unrounded + 0.5);
                        } else {
                            q_rounded = @ceil(q_unrounded - 0.5);
                        }

                        const q_clamped = std.math.clamp(q_rounded, q_min, q_max);
                        output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                    }
                }
            }
        }
    }
}
