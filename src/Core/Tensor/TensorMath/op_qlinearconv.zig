const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("op_convolution.zig");

// HELPER FUNCTIONS FOR CORRECT QUANTIZATION
inline fn readScalarZP(comptime T: type, zp_any: anytype) i32 {
    _ = T;

    // Simple approach: check if it's a pointer first
    const ZPType = @TypeOf(zp_any);
    if (@typeInfo(ZPType) == .pointer) {
        // Dereference and try to access .data[0]
        return @as(i32, @intCast(zp_any.data[0]));
    } else {
        // It's already a scalar or struct, use directly
        return @as(i32, @intCast(zp_any));
    }
}

inline fn readPerChannelScale(comptime T: type, s: *const Tensor(T), m: usize, M: usize) f32 {
    if (s.shape.len == 1 and s.shape[0] == M) return @as(f32, @floatCast(s.data[m]));
    return @as(f32, @floatCast(s.data[0])); // broadcast
}

inline fn readPerChannelZP(comptime T: type, zp_any: anytype, m: usize, M: usize) i32 {
    return switch (@TypeOf(zp_any)) {
        *const Tensor(T) => {
            if (zp_any.shape.len == 1 and zp_any.shape[0] == M) return @as(i32, @intCast(zp_any.data[m]));
            return @as(i32, @intCast(zp_any.data[0]));
        },
        Tensor(T) => {
            if (zp_any.shape.len == 1 and zp_any.shape[0] == M) return @as(i32, @intCast(zp_any.data[0]));
            return @as(i32, @intCast(zp_any.data[0]));
        },
        T => @as(i32, @intCast(zp_any)),
        else => @compileError("zp per-channel expected"),
    };
}

const SHIFT: u5 = 16;
inline fn q16(x: f32) i32 {
    return @as(i32, @intFromFloat(x * @as(f32, @floatFromInt(@as(u32, 1) << SHIFT))));
}

inline fn rshift_round_s64(x: i64, comptime shift_bits: u5) i64 {
    const bias: i64 = if (x >= 0) (1 << (shift_bits - 1)) else -(1 << (shift_bits - 1));
    return (x + bias) >> shift_bits;
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
pub fn qlinearconv(
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

    // Perform quantized convolution
    try qlinearconv_lean(InputType, WeightType, ScaleType, ZeroPointType, BiasType, input_ptr, x_scale, x_zero_point, w, w_scale, w_zero_point, &output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad.?);

    return output;
}

/// OTTIMIZZATO: Lean version of QLinearConv with pre-computation and cache optimizations
pub fn qlinearconv_lean(
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
    _ = auto_pad; // non gestito: usare pads espliciti

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

    // Estrai dimensioni
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

    // Parametri
    const actual_group = group orelse 1;
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Gruppi validation
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

    // ===== OTTIMIZZAZIONE 1: Pre-calcolo scale e bias =====
    const x_scale_val: f32 = asF32(ScaleType, x_scale.data[0]);
    const y_scale_val: f32 = asF32(ScaleType, y_scale.data[0]);
    const x_zp_f: f32 = if (x_zero_point.data.len > 0) asF32(@TypeOf(x_zero_point.data[0]), x_zero_point.data[0]) else 0.0;
    const y_zp_f: f32 = if (y_zero_point.data.len > 0) asF32(@TypeOf(y_zero_point.data[0]), y_zero_point.data[0]) else 0.0;

    // Pre-calcola scale e bias per tutti i canali output (evita calcoli ridondanti)
    var channel_scales = std.ArrayList(f32).init(pkg_allocator);
    defer channel_scales.deinit();
    var channel_zps = std.ArrayList(f32).init(pkg_allocator);
    defer channel_zps.deinit();
    var channel_bias = std.ArrayList(f32).init(pkg_allocator);
    defer channel_bias.deinit();

    try channel_scales.ensureTotalCapacity(out_channels);
    try channel_zps.ensureTotalCapacity(out_channels);
    try channel_bias.ensureTotalCapacity(out_channels);

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

    // Pre-calcola i limiti di quantizzazione
    const q_min: f32 = asF32(InputType, std.math.minInt(InputType));
    const q_max: f32 = asF32(InputType, std.math.maxInt(InputType));

    // ===== OTTIMIZZAZIONE 2: Specialized paths per kernel comuni =====
    if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
        // Ottimizzato per 3x3 (MobileNet style)
        try conv3x3Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else if (kernel_height == 1 and kernel_width == 1) {
        // Ottimizzato per 1x1 (pointwise)
        try conv1x1Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else {
        // ===== OTTIMIZZAZIONE 3: Loop originale con pre-calcoli =====
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
                                const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
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

// ===== SPECIALIZZAZIONI OTTIMIZZATE =====

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

                        // Unroll 3x3 kernel manually per migliori performance
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
                            // Unroll rimanenti per brevità...
                            inline for (0..3) |kh| {
                                inline for (0..3) |kw| {
                                    if (kh == 0 and kw == 0) continue; // già fatto sopra

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
                        const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
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

                // 1x1 conv è matrix multiplication - ottimizzato di conseguenza
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var acc: f32 = bias_f;

                        // Nessun kernel spaziale, solo channel mixing
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
                        const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
                        output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                    }
                }
            }
        }
    }
}

/// Embedded-optimized version using fixed-point arithmetic (Q15.16)
/// Reduces floating-point operations for better performance on embedded targets
pub fn qlinearconv_embedded_lean(
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
    _ = auto_pad;

    // Debug: basic shapes

    // Check tensor shapes
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
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
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    const in_height = x.shape[2];
    const in_width = x.shape[3];

    const out_channels = w.shape[0];
    const weight_in_channels = w.shape[1];
    const kernel_height = w.shape[2];
    const kernel_width = w.shape[3];

    const out_height = output.shape[2];
    const out_width = output.shape[3];

    // Default stride, pads, dilations
    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;

    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h_begin = pads_arr[0];
    const pad_w_begin = pads_arr[1];

    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    const actual_group = group orelse 1;

    // Debug: parameters

    // Validate groups and channel divisibility
    if (in_channels % actual_group != 0 or out_channels % actual_group != 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * actual_group != in_channels) {
        return TensorMathError.InvalidDimensions;
    }

    // Pre-compute common values once
    const x_scale_val: f32 = asF32(ScaleType, x_scale.data[0]);
    const x_zp_f: f32 = if (x_zero_point.data.len > 0) asF32(@TypeOf(x_zero_point.data[0]), x_zero_point.data[0]) else 0.0;
    const y_scale_val: f32 = asF32(ScaleType, y_scale.data[0]);
    const y_zp_f: f32 = if (y_zero_point.data.len > 0) asF32(@TypeOf(y_zero_point.data[0]), y_zero_point.data[0]) else 0.0;

    // DEBUG: Print scale factors for first call
    const debug_first_call = (batch_size == 1 and in_channels == 3 and in_height == 96);
    if (debug_first_call) {
        // std.debug.print("[CONV_DEBUG] Scale factors:\n", .{});
        // std.debug.print("  x_scale_val = {d}\n", .{x_scale_val});
        // std.debug.print("  x_zp_f = {d}\n", .{x_zp_f});
        // std.debug.print("  y_scale_val = {d}\n", .{y_scale_val});
        // std.debug.print("  y_zp_f = {d}\n", .{y_zp_f});
        // std.debug.print("  y_scale_inv = {d}\n", .{1.0 / y_scale_val});
    }

    // Fixed-point scale factors (Q15.16 format - correct for proper precision)
    const SCALE_SHIFT: u5 = 16;
    const x_scale_fixed: i32 = @as(i32, @intFromFloat(x_scale_val * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));
    const x_zp_fixed: i32 = @as(i32, @intFromFloat(x_zp_f * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));
    const y_scale_inv_fixed: i32 = @as(i32, @intFromFloat((1.0 / y_scale_val) * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));
    const y_zp_fixed: i32 = @as(i32, @intFromFloat(y_zp_f * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));

    if (debug_first_call) {
        // std.debug.print("[CONV_DEBUG] Fixed-point factors:\n", .{});
        // std.debug.print("  x_scale_fixed = {d}\n", .{x_scale_fixed});
        // std.debug.print("  x_zp_fixed = {d}\n", .{x_zp_fixed});
        // std.debug.print("  y_scale_inv_fixed = {d}\n", .{y_scale_inv_fixed});
        // std.debug.print("  y_zp_fixed = {d}\n", .{y_zp_fixed});
        // std.debug.print("  SCALE_SHIFT = {d}\n", .{SCALE_SHIFT});
    }

    const q_min: i32 = std.math.minInt(InputType);
    const q_max: i32 = std.math.maxInt(InputType);

    const expected_output_len = batch_size * out_channels * out_height * out_width;
    if (expected_output_len != output.data.len) {}

    // Main computation loops - optimized for cache efficiency
    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                // Per-channel weight scaling
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

                const w_scale_fixed: i32 = @as(i32, @intFromFloat(w_scale_val * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));
                const w_zp_fixed: i32 = @as(i32, @intFromFloat(w_zp_f * @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))));

                // FIX: bias nella stessa scala dell'accumulatore (Q16)
                const bias_contribution: i64 = if (bias) |b| blk: {
                    const b_raw = if (b.data.len == 1) b.data[0] else b.data[m];
                    var b_real: f32 = 0.0;
                    switch (@typeInfo(BiasType)) {
                        .int => { // bias i32 quantizzato ONNX: è nel dominio accumulatore (int), lo porto in reale
                            b_real = asF32(BiasType, b_raw) * (x_scale_val * w_scale_val);
                        },
                        .float => { // bias float: è già reale
                            b_real = asF32(BiasType, b_raw);
                        },
                        else => {
                            b_real = asF32(BiasType, b_raw); // prudente
                        },
                    }
                    // b_real è float nel dominio reale, ora portalo in Q16
                    const bias_q16: i64 = @as(i64, @intFromFloat(@round(b_real * (@as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT))))));
                    break :blk bias_q16;
                } else 0;

                // Debug per-channel
                if (m == 0 and n == 0) {}

                // Specialized paths for common kernel sizes (embedded optimization)
                if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
                    // Debug for 3x3 conv (DISABLED)
                    // Optimized 3x3 convolution with loop unrolling
                    conv3x3EmbeddedOptimized(x, w, output, n, m, in_c_start, in_c_end, in_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_fixed, x_zp_fixed, w_scale_fixed, w_zp_fixed, y_scale_inv_fixed, y_zp_fixed, bias_contribution, q_min, q_max, SCALE_SHIFT, InputType, WeightType, debug_first_call, x_scale_val, w_scale_val, y_scale_val, x_zero_point, w_zero_point);
                } else if (kernel_height == 1 and kernel_width == 1) {
                    // Optimized 1x1 convolution (pointwise)
                    conv1x1EmbeddedOptimized(x, w, output, n, m, in_c_start, in_c_end, in_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_fixed, x_zp_fixed, w_scale_fixed, w_zp_fixed, y_scale_inv_fixed, y_zp_fixed, bias_contribution, q_min, q_max, SCALE_SHIFT, InputType, WeightType, x_zero_point, w_zero_point);
                } else {
                    // Generic case with fixed-point optimizations (inline to avoid comptime issues)
                    for (0..out_height) |oh| {
                        const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                        for (0..out_width) |ow| {
                            const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));

                            var acc: i64 = bias_contribution; // acc è ora in Q16

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

                                        const x_q = @as(i32, @intCast(x.data[input_idx]));
                                        const w_q = @as(i32, @intCast(w.data[weight_idx]));

                                        // FIXED: Schema Q16 coerente con gli altri path
                                        const Zx = readScalarZP(InputType, x_zero_point);
                                        const Zw = readScalarZP(WeightType, w_zero_point);
                                        const x_q16 = @as(i64, x_q - Zx) * @as(i64, x_scale_fixed);
                                        const w_q16 = @as(i64, w_q - Zw) * @as(i64, w_scale_fixed);
                                        const product_q16 = (x_q16 * w_q16) >> SCALE_SHIFT; // Q16

                                        // DEBUG: First few calculations only
                                        if (debug_first_call and n == 0 and m == 0 and oh == 0 and ow == 0 and kh == 0 and kw == 0 and c < 3) {
                                            // std.debug.print("[CONV_DEBUG] Calc[c={d}]: x_q={d}, w_q={d}\n", .{ c, x_q, w_q });
                                            // std.debug.print("  x_q16={d}, w_q16={d}\n", .{ x_q16, w_q16 });
                                            // std.debug.print("  product_q16={d}, acc_before={d}\n", .{ product_q16, acc });
                                        }

                                        acc += product_q16; // acc resta Q16

                                        if (debug_first_call and n == 0 and m == 0 and oh == 0 and ow == 0 and kh == 0 and kw == 0 and c < 3) {
                                            // std.debug.print("  acc_after={d}\n", .{acc});
                                        }
                                    }
                                }
                            }

                            // acc è in Q16
                            const t_q16: i64 = (acc * @as(i64, y_scale_inv_fixed)) >> SCALE_SHIFT; // ancora Q16
                            const with_zp_q16: i64 = t_q16 + @as(i64, y_zp_fixed); // ancora Q16

                            // round-to-nearest e ritorno a Q0
                            const rounding: i64 = @as(i64, 1) << (SCALE_SHIFT - 1);
                            const q_q0: i32 = @as(i32, @intCast((with_zp_q16 + rounding) >> SCALE_SHIFT));

                            // clamp nel range del tipo di uscita
                            const q_clamped = std.math.clamp(q_q0, q_min, q_max);

                            // DEBUG: Final quantization step for first few outputs
                            if (debug_first_call and n == 0 and m == 0 and oh == 0 and ow < 3) {
                                // std.debug.print("[CONV_DEBUG] Final[oh={d},ow={d}]: acc={d} (Q16)\n", .{ oh, ow, acc });
                                // std.debug.print("  t_q16={d}, with_zp_q16={d}, q_q0={d}\n", .{ t_q16, with_zp_q16, q_q0 });
                                // std.debug.print("  q_min={d}, q_max={d}, q_clamped={d}\n", .{ q_min, q_max, q_clamped });
                                if (q_q0 != q_clamped) {
                                    // std.debug.print("  *** CLAMPING OCCURRED! ***\n", .{});
                                }
                            }

                            const output_idx = ((n * output.shape[1] + m) * out_height + oh) * out_width + ow;
                            if (output_idx >= output.data.len) {
                                return TensorError.IndexOutOfBounds;
                            }
                            output.data[output_idx] = @as(InputType, @intCast(q_clamped));
                        }
                    }
                }
            }
        }
    }
}

// Optimized 3x3 convolution with manual loop unrolling (embedded version)
inline fn conv3x3EmbeddedOptimized(x: anytype, w: anytype, output: anytype, n: usize, m: usize, in_c_start: usize, in_c_end: usize, in_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, stride_h: usize, stride_w: usize, pad_h_begin: usize, pad_w_begin: usize, x_scale_fixed: i32, x_zp_fixed: i32, w_scale_fixed: i32, w_zp_fixed: i32, y_scale_inv_fixed: i32, y_zp_fixed: i32, bias_contribution: i64, q_min: i32, q_max: i32, scale_shift: u5, comptime InputType: type, comptime WeightType: type, debug: bool, x_scale_val: f32, w_scale_val: f32, y_scale_val: f32, x_zero_point: anytype, w_zero_point: anytype) void {
    _ = x_scale_val;
    _ = w_scale_val;
    _ = y_scale_val;
    _ = x_zp_fixed;
    _ = w_zp_fixed;
    for (0..out_height) |oh| {
        const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

        for (0..out_width) |ow| {
            const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));

            var acc: i64 = bias_contribution;

            // DEBUG: For first few pixels
            if (debug and n == 0 and m == 0 and oh < 2 and ow < 3) {
                // std.debug.print("[CONV_DEBUG] 3x3 FULL KERNEL convolution at pixel [{d},{d}]:\n", .{ oh, ow });
                // std.debug.print("  bias_contribution = {d} (Q16)\n", .{bias_contribution});
                // std.debug.print("  in_h_start = {d}, in_w_start = {d}\n", .{ in_h_start, in_w_start });
                // std.debug.print("  Using ALL 9 TAPS of 3x3 kernel!\n", .{});
            }

            // Unrolled 3x3 kernel - manual unrolling for embedded performance
            // kh = 0
            const in_h_0 = in_h_start;
            if (in_h_0 >= 0 and in_h_0 < @as(isize, @intCast(in_height))) {
                const ih_0 = @as(usize, @intCast(in_h_0));

                // kw = 0
                const in_w_0 = in_w_start;
                if (in_w_0 >= 0 and in_w_0 < @as(isize, @intCast(in_width))) {
                    const iw_0 = @as(usize, @intCast(in_w_0));
                    for (in_c_start..in_c_end) |c| {
                        const k_c = c - in_c_start;
                        const input_idx = ((n * in_channels + c) * in_height + ih_0) * in_width + iw_0;
                        const weight_idx = ((m * weight_in_channels + k_c) * 3 + 0) * 3 + 0;

                        const x_q = @as(i32, @intCast(x.data[input_idx]));
                        const w_q = @as(i32, @intCast(w.data[weight_idx]));

                        // FIX: Formula corretta - zero-point NON è già scalato
                        const Zx_i32 = readScalarZP(InputType, x_zero_point);
                        const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                        const x_diff = x_q - Zx_i32;
                        const w_diff = w_q - Zw_i32;

                        // FIXED: Mantieni Q16 fino al termine del pixel
                        const x_q16 = @as(i64, x_diff) * @as(i64, x_scale_fixed); // Q16
                        const w_q16 = @as(i64, w_diff) * @as(i64, w_scale_fixed); // Q16

                        // prodotto in Q16 (Q16*Q16 = Q32, >>16 -> Q16)
                        const product_q16 = (x_q16 * w_q16) >> scale_shift; // Q16

                        // DEBUG for first few calculations when inside bounds
                        if (debug and n == 0 and m == 0 and oh < 2 and ow < 3 and c < 2) {
                            // std.debug.print("[CONV_DEBUG] [{d},{d}] kh=0,kw=0,c={d}: x_q={d}, w_q={d}\n", .{ oh, ow, c, x_q, w_q });
                            // std.debug.print("  x_q16={d}, w_q16={d}, product_q16={d}\n", .{ x_q16, w_q16, product_q16 });
                            // std.debug.print("  acc_before={d}\n", .{acc});
                        }

                        acc += product_q16; // acc è ora in Q16

                        if (debug and n == 0 and m == 0 and oh < 2 and ow < 3 and c < 2 and product_q16 != 0) {
                            // std.debug.print("  NON-ZERO contribution! acc_after={d}\n", .{acc});
                        }
                    }
                }

                // FIXED: Implementiamo tutti i 9 tap del kernel 3x3!
                // kw = 1
                const in_w_1 = in_w_start + 1;
                if (in_w_1 >= 0 and in_w_1 < @as(isize, @intCast(in_width))) {
                    const iw_1 = @as(usize, @intCast(in_w_1));
                    for (in_c_start..in_c_end) |c| {
                        const k_c = c - in_c_start;
                        const input_idx = ((n * in_channels + c) * in_height + ih_0) * in_width + iw_1;
                        const weight_idx = ((m * weight_in_channels + k_c) * 3 + 0) * 3 + 1;

                        const x_q = @as(i32, @intCast(x.data[input_idx]));
                        const w_q = @as(i32, @intCast(w.data[weight_idx]));

                        const Zx_i32 = readScalarZP(InputType, x_zero_point);
                        const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                        const x_q16 = @as(i64, x_q - Zx_i32) * @as(i64, x_scale_fixed);
                        const w_q16 = @as(i64, w_q - Zw_i32) * @as(i64, w_scale_fixed);
                        acc += (x_q16 * w_q16) >> scale_shift;
                    }
                }

                // kw = 2
                const in_w_2 = in_w_start + 2;
                if (in_w_2 >= 0 and in_w_2 < @as(isize, @intCast(in_width))) {
                    const iw_2 = @as(usize, @intCast(in_w_2));
                    for (in_c_start..in_c_end) |c| {
                        const k_c = c - in_c_start;
                        const input_idx = ((n * in_channels + c) * in_height + ih_0) * in_width + iw_2;
                        const weight_idx = ((m * weight_in_channels + k_c) * 3 + 0) * 3 + 2;

                        const x_q = @as(i32, @intCast(x.data[input_idx]));
                        const w_q = @as(i32, @intCast(w.data[weight_idx]));

                        const Zx_i32 = readScalarZP(InputType, x_zero_point);
                        const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                        const x_q16 = @as(i64, x_q - Zx_i32) * @as(i64, x_scale_fixed);
                        const w_q16 = @as(i64, w_q - Zw_i32) * @as(i64, w_scale_fixed);
                        acc += (x_q16 * w_q16) >> scale_shift;
                    }
                }
            }

            // kh = 1 (tutte le kw)
            const in_h_1 = in_h_start + 1;
            if (in_h_1 >= 0 and in_h_1 < @as(isize, @intCast(in_height))) {
                const ih_1 = @as(usize, @intCast(in_h_1));
                for (0..3) |kw| {
                    const in_w = in_w_start + @as(isize, @intCast(kw));
                    if (in_w >= 0 and in_w < @as(isize, @intCast(in_width))) {
                        const iw = @as(usize, @intCast(in_w));
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;
                            const input_idx = ((n * in_channels + c) * in_height + ih_1) * in_width + iw;
                            const weight_idx = ((m * weight_in_channels + k_c) * 3 + 1) * 3 + kw;

                            const x_q = @as(i32, @intCast(x.data[input_idx]));
                            const w_q = @as(i32, @intCast(w.data[weight_idx]));

                            const Zx_i32 = readScalarZP(InputType, x_zero_point);
                            const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                            const x_q16 = @as(i64, x_q - Zx_i32) * @as(i64, x_scale_fixed);
                            const w_q16 = @as(i64, w_q - Zw_i32) * @as(i64, w_scale_fixed);
                            acc += (x_q16 * w_q16) >> scale_shift;
                        }
                    }
                }
            }

            // kh = 2 (tutte le kw)
            const in_h_2 = in_h_start + 2;
            if (in_h_2 >= 0 and in_h_2 < @as(isize, @intCast(in_height))) {
                const ih_2 = @as(usize, @intCast(in_h_2));
                for (0..3) |kw| {
                    const in_w = in_w_start + @as(isize, @intCast(kw));
                    if (in_w >= 0 and in_w < @as(isize, @intCast(in_width))) {
                        const iw = @as(usize, @intCast(in_w));
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;
                            const input_idx = ((n * in_channels + c) * in_height + ih_2) * in_width + iw;
                            const weight_idx = ((m * weight_in_channels + k_c) * 3 + 2) * 3 + kw;

                            const x_q = @as(i32, @intCast(x.data[input_idx]));
                            const w_q = @as(i32, @intCast(w.data[weight_idx]));

                            const Zx_i32 = readScalarZP(InputType, x_zero_point);
                            const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                            const x_q16 = @as(i64, x_q - Zx_i32) * @as(i64, x_scale_fixed);
                            const w_q16 = @as(i64, w_q - Zw_i32) * @as(i64, w_scale_fixed);
                            acc += (x_q16 * w_q16) >> scale_shift;
                        }
                    }
                }
            }

            // acc è in Q16
            const t_q16: i64 = (acc * @as(i64, y_scale_inv_fixed)) >> scale_shift; // ancora Q16
            const with_zp_q16: i64 = t_q16 + @as(i64, y_zp_fixed); // ancora Q16

            // round-to-nearest e ritorno a Q0
            const rounding: i64 = @as(i64, 1) << (scale_shift - 1);
            const q_q0: i32 = @as(i32, @intCast((with_zp_q16 + rounding) >> scale_shift));

            // clamp nel range del tipo di uscita
            const q_clamped = std.math.clamp(q_q0, q_min, q_max);

            // DEBUG: Final quantization for first few pixels
            if (debug and n == 0 and m == 0 and oh == 0 and ow < 3) {
                // std.debug.print("[CONV_DEBUG] 3x3 FULL KERNEL Final[oh={d},ow={d}]: final_acc={d} (Q16)\n", .{ oh, ow, acc });
                // std.debug.print("  t_q16={d}, with_zp={d}, q_q0={d}, q_clamped={d}\n", .{ t_q16, with_zp_q16, q_q0, q_clamped });
                if (q_q0 != q_clamped) {
                    // std.debug.print("  *** CLAMPING OCCURRED! ***\n", .{});
                }
            }

            const output_idx = ((n * output.shape[1] + m) * out_height + oh) * out_width + ow;
            output.data[output_idx] = @as(InputType, @intCast(q_clamped));
        }
    }
}

// Optimized 1x1 convolution (pointwise) (embedded version)
inline fn conv1x1EmbeddedOptimized(x: anytype, w: anytype, output: anytype, n: usize, m: usize, in_c_start: usize, in_c_end: usize, in_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, stride_h: usize, stride_w: usize, pad_h_begin: usize, pad_w_begin: usize, x_scale_fixed: i32, x_zp_fixed: i32, w_scale_fixed: i32, w_zp_fixed: i32, y_scale_inv_fixed: i32, y_zp_fixed: i32, bias_contribution: i64, q_min: i32, q_max: i32, scale_shift: u5, comptime InputType: type, comptime WeightType: type, x_zero_point: anytype, w_zero_point: anytype) void {
    _ = x_zp_fixed;
    _ = w_zp_fixed;
    // FIXED: 1x1 conv with proper stride/padding handling
    for (0..out_height) |oh| {
        const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));
        for (0..out_width) |ow| {
            const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));
            var acc: i64 = bias_contribution;

            // Check bounds for 1x1 kernel
            if (in_h_start >= 0 and in_h_start < @as(isize, @intCast(in_height)) and
                in_w_start >= 0 and in_w_start < @as(isize, @intCast(in_width)))
            {
                const ih = @as(usize, @intCast(in_h_start));
                const iw = @as(usize, @intCast(in_w_start));

                for (in_c_start..in_c_end) |c| {
                    const k_c = c - in_c_start;
                    const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                    const weight_idx = m * weight_in_channels + k_c;

                    const x_q = @as(i32, @intCast(x.data[input_idx]));
                    const w_q = @as(i32, @intCast(w.data[weight_idx]));

                    // FIX: Formula corretta - zero-point NON è già scalato
                    const Zx_i32 = readScalarZP(InputType, x_zero_point);
                    const Zw_i32 = readScalarZP(WeightType, w_zero_point);
                    const x_diff = x_q - Zx_i32;
                    const w_diff = w_q - Zw_i32;

                    // FIXED: Mantieni Q16 fino al termine del pixel
                    const x_q16 = @as(i64, x_diff) * @as(i64, x_scale_fixed); // Q16
                    const w_q16 = @as(i64, w_diff) * @as(i64, w_scale_fixed); // Q16

                    // prodotto in Q16 (Q16*Q16 = Q32, >>16 -> Q16)
                    const product_q16 = (x_q16 * w_q16) >> scale_shift; // Q16
                    acc += product_q16; // acc è ora in Q16
                }
            }

            // acc è in Q16
            const t_q16: i64 = (acc * @as(i64, y_scale_inv_fixed)) >> scale_shift; // ancora Q16
            const with_zp_q16: i64 = t_q16 + @as(i64, y_zp_fixed); // ancora Q16

            // round-to-nearest e ritorno a Q0
            const rounding: i64 = @as(i64, 1) << (scale_shift - 1);
            const q_q0: i32 = @as(i32, @intCast((with_zp_q16 + rounding) >> scale_shift));

            // clamp nel range del tipo di uscita
            const q_clamped = std.math.clamp(q_q0, q_min, q_max);

            const output_idx = ((n * output.shape[1] + m) * out_height + oh) * out_width + ow;
            output.data[output_idx] = @as(InputType, @intCast(q_clamped));
        }
    }
}

// Generic convolution with fixed-point optimizations
inline fn convGenericOptimized(x: anytype, w: anytype, output: anytype, n: usize, m: usize, in_c_start: usize, in_c_end: usize, in_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, kernel_height: usize, kernel_width: usize, stride_h: usize, stride_w: usize, pad_h_begin: usize, pad_w_begin: usize, dilation_h: usize, dilation_w: usize, x_scale_fixed: i32, x_zp_fixed: i32, w_scale_fixed: i32, w_zp_fixed: i32, y_scale_inv_fixed: i32, y_zp_fixed: i32, bias_contribution: i64, q_min: i32, q_max: i32, scale_shift: u5, comptime InputType: type, comptime WeightType: type) void {
    _ = WeightType;
    for (0..out_height) |oh| {
        const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

        for (0..out_width) |ow| {
            const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));

            var acc: i64 = bias_contribution;

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

                        const x_q = @as(i32, @intCast(x.data[input_idx]));
                        const w_q = @as(i32, @intCast(w.data[weight_idx]));

                        const x_dequant = (x_q * x_scale_fixed - x_zp_fixed * x_scale_fixed) >> scale_shift;
                        const w_dequant = (w_q * w_scale_fixed - w_zp_fixed * w_scale_fixed) >> scale_shift;

                        acc += @as(i64, x_dequant) * @as(i64, w_dequant);
                    }
                }
            }

            const acc_scaled = (acc * @as(i64, y_scale_inv_fixed)) >> scale_shift;
            const q_result = @as(i32, @intCast(acc_scaled)) + (y_zp_fixed >> scale_shift);
            const q_clamped = std.math.clamp(q_result, q_min, q_max);

            const output_idx = ((n * output.shape[1] + m) * out_height + oh) * out_width + ow;
            output.data[output_idx] = @as(InputType, @intCast(q_clamped));
        }
    }
}

/// Calculate output shape for QLinearConv - same as regular Conv
pub fn get_qlinearconv_output_shape(
    comptime T: type,
    input_shape: []const usize,
    weight_shape: []const usize,
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    auto_pad: ?[]const u8,
) ![]usize {
    return conv.calculateOutputShape(T, input_shape, weight_shape, stride, pads, dilations, auto_pad);
}
