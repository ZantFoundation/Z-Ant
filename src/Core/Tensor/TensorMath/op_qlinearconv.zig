const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("op_convolution.zig");

// HELPER FUNCTIONS FOR CORRECT QUANTIZATION
inline fn readScalarZP(comptime T: type, zp_any: anytype) i32 {
    _ = T;
    return readScalarZPInternal(zp_any);
}

fn readScalarZPInternal(zp_any: anytype) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readScalarZPInternal(zp_any.*),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                break :blk @as(i32, @intCast(zp_any[0]));
            },
            .many, .c => blk: {
                // Treat bare pointers as single-value buffers and read the first element.
                break :blk @as(i32, @intCast(zp_any[0]));
            },
        },
        .optional => if (zp_any) |payload| readScalarZPInternal(payload) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(data[0]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

inline fn readPerChannelScale(comptime T: type, s: *const Tensor(T), m: usize, M: usize) f32 {
    if (s.shape.len == 1 and s.shape[0] == M) return @as(f32, @floatCast(s.data[m]));
    return @as(f32, @floatCast(s.data[0])); // broadcast
}

inline fn selectChannelIndex(len: usize, channel: usize) usize {
    if (len <= 1) return 0;
    return if (channel < len) channel else len - 1;
}

inline fn readPerChannelZP(zp_any: anytype, m: usize, M: usize) i32 {
    _ = M;
    return readPerChannelZPInternal(zp_any, m);
}

fn readPerChannelZPInternal(zp_any: anytype, m: usize) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readPerChannelZPInternal(zp_any.*, m),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                const idx = selectChannelIndex(zp_any.len, m);
                break :blk @as(i32, @intCast(zp_any[idx]));
            },
            .many, .c => @compileError("unsupported zero-point pointer representation"),
        },
        .optional => if (zp_any) |payload| readPerChannelZPInternal(payload, m) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.array.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.vector.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            const idx = selectChannelIndex(data.len, m);
            break :blk @as(i32, @intCast(data[idx]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

fn applyAsymmetricPadding(
    comptime InputType: type,
    original: *const Tensor(InputType),
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    fill_zero_point: i32,
) !Tensor(InputType) {
    const batch = original.shape[0];
    const channels = original.shape[1];
    const in_height = original.shape[2];
    const in_width = original.shape[3];

    const padded_height = in_height + pad_top + pad_bottom;
    const padded_width = in_width + pad_left + pad_right;
    var padded_shape = [_]usize{ batch, channels, padded_height, padded_width };

    var padded = try Tensor(InputType).fromShape(&pkg_allocator, &padded_shape);
    errdefer padded.deinit();

    const min_val = std.math.minInt(InputType);
    const max_val = std.math.maxInt(InputType);
    const fill_value = @as(InputType, @intCast(std.math.clamp(fill_zero_point, min_val, max_val)));
    for (padded.data) |*elem| {
        elem.* = fill_value;
    }

    const src_channel_stride = in_height * in_width;
    const dst_channel_stride = padded_height * padded_width;
    const src_batch_stride = channels * src_channel_stride;
    const dst_batch_stride = channels * dst_channel_stride;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const src_channel_base = n * src_batch_stride + c * src_channel_stride;
            const dst_channel_base = n * dst_batch_stride + c * dst_channel_stride;

            var h: usize = 0;
            while (h < in_height) : (h += 1) {
                const src_row = src_channel_base + h * in_width;
                const dst_row = dst_channel_base + (h + pad_top) * padded_width + pad_left;
                std.mem.copyForwards(InputType, padded.data[dst_row .. dst_row + in_width], original.data[src_row .. src_row + in_width]);
            }
        }
    }

    return padded;
}

const SHIFT: u5 = 16;
inline fn q16(x: f32) i32 {
    const scaled = x * @as(f32, @floatFromInt(@as(u32, 1) << SHIFT));
    // Clamp to i32 bounds to prevent overflow
    if (scaled > @as(f32, @floatFromInt(std.math.maxInt(i32)))) {
        return std.math.maxInt(i32);
    }
    if (scaled < @as(f32, @floatFromInt(std.math.minInt(i32)))) {
        return std.math.minInt(i32);
    }
    return @as(i32, @intFromFloat(scaled));
}

inline fn rshift_round_s64(x: i64, comptime shift_bits: u5) i64 {
    const bias: i64 = if (x >= 0) (1 << (shift_bits - 1)) else -(1 << (shift_bits - 1));
    return (x + bias) >> shift_bits;
}

inline fn clampToI8(value: i32) i8 {
    var v = value;
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return @as(i8, @intCast(v));
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
    const output_shape = try conv.calculateOutputShape(InputType, pkg_allocator, input_shape[0..], w.shape, stride, pads, dilations, auto_pad);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized convolution using dispatch to get the best implementation
    try qlinearconv_dispatch(InputType, WeightType, ScaleType, InputType, BiasType, input_ptr, x_scale, x_zero_point, w, w_scale, w_zero_point, &output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad.?);

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
    // DEBUG: Print which function is being called
    // std.debug.print("QLINEAR_DEBUG: using qlinearconv_lean (floating point)\n", .{});
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

const ChannelParams = struct {
    combined_scale_q16: i64,
    weight_zero_point: i32,
    bias_q16: i64,
};

const QuantParams = struct {
    scale_shift: u5,
    input_zero_point: i32,
    output_inv_scale_i64: i64,
    output_zero_point_q16: i64,
    q_min: i32,
    q_max: i32,
};

inline fn saturateI128ToI64(value: i128) i64 {
    const max_i64_as_i128 = @as(i128, std.math.maxInt(i64));
    const min_i64_as_i128 = @as(i128, std.math.minInt(i64));

    if (value > max_i64_as_i128) {
        return std.math.maxInt(i64);
    }
    if (value < min_i64_as_i128) {
        return std.math.minInt(i64);
    }
    return @intCast(value);
}

const ConvDims = struct {
    batch: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    group_in_channels: usize,
    group_out_channels: usize,
};

const ConvLayout = struct {
    input_batch_stride: usize,
    input_channel_stride: usize,
    input_row_stride: usize,
    weight_out_stride: usize,
    weight_channel_stride: usize,
    weight_row_stride: usize,
    output_batch_stride: usize,
    output_channel_stride: usize,
    output_row_stride: usize,
};

inline fn quantizeAccumulator(acc: i64, quant: QuantParams) i32 {
    // Step 1: scale by 1/y_scale keeping Q16
    const tmp_q16 = (acc * quant.output_inv_scale_i64) >> @as(u6, @intCast(quant.scale_shift));
    // Step 2: add zero point in Q16
    const with_zp_q16 = tmp_q16 + quant.output_zero_point_q16;

    // Step 3: round-to-nearest-even from Q16 to integer
    const denom: i64 = @as(i64, 1) << @as(u6, @intCast(quant.scale_shift));
    const half: i64 = denom >> 1;
    var q64 = @divTrunc(with_zp_q16, denom); // toward zero
    const rem = with_zp_q16 - q64 * denom; // remainder with sign of with_zp_q16
    const sign: i64 = if (with_zp_q16 >= 0) 1 else -1;
    const abs_rem = if (rem >= 0) rem else -rem;

    if (abs_rem > half) {
        q64 += sign;
    } else if (abs_rem == half) {
        if ((q64 & 1) != 0) q64 += sign;
    }

    var q = @as(i32, @intCast(q64));
    if (q < quant.q_min) q = quant.q_min;
    if (q > quant.q_max) q = quant.q_max;
    return q;
}

/// Embedded-optimized version using fixed-point arithmetic (Q15.16)
/// Reduces floating-point operations for better performance on embedded targets
pub inline fn qlinearconv_embedded_lean(
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
    // DEBUG: Print which function is being called
    // std.debug.print("QLINEAR_DEBUG: using qlinearconv_embedded_lean, input[0]={}\n", .{x.data[0]});
    if (auto_pad.len != 0 and !std.mem.eql(u8, auto_pad, "NOTSET")) {
        return TensorMathError.InvalidPadding;
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

    if (!isInt(InputType) or !isInt(WeightType)) {
        // DEBUG: fallback to floating-point
        // std.debug.print("QLINEAR_DEBUG: embedded_lean fallback to qlinearconv_lean because InputType={s} isInt={}\n", .{ @typeName(InputType), isInt(InputType) });
        return qlinearconv_lean(InputType, WeightType, ScaleType, void, BiasType, x, x_scale, x_zero_point, w, w_scale, w_zero_point, output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad);
    }

    // Pure reference implementation - no CMSIS dispatch overhead

    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

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

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h_begin = pads_arr[0];
    const pad_w_begin = pads_arr[1];
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;
    const actual_group = group orelse 1;

    if (in_channels % actual_group != 0 or out_channels % actual_group != 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * actual_group != in_channels) {
        return TensorMathError.InvalidDimensions;
    }

    const SCALE_SHIFT: u5 = 16;
    const scale_factor = @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT));
    const x_scale_val = asF32(ScaleType, x_scale.data[0]);
    const x_scale_q16 = @as(i64, q16(x_scale_val));
    const y_scale_val = asF32(ScaleType, y_scale.data[0]);
    const input_zero_point = if (@typeInfo(@TypeOf(x_zero_point)) == .pointer and x_zero_point.data.len == 0)
        0
    else
        readScalarZP(InputType, x_zero_point);
    const output_zero_point = if (@typeInfo(@TypeOf(y_zero_point)) == .pointer and y_zero_point.data.len == 0)
        0
    else
        readScalarZP(InputType, y_zero_point);

    const quant = QuantParams{
        .scale_shift = SCALE_SHIFT,
        .input_zero_point = input_zero_point,
        .output_inv_scale_i64 = @as(i64, q16(1.0 / y_scale_val)),
        .output_zero_point_q16 = @as(i64, output_zero_point) << SCALE_SHIFT,
        .q_min = @as(i32, @intCast(std.math.minInt(InputType))),
        .q_max = @as(i32, @intCast(std.math.maxInt(InputType))),
    };

    const dims = ConvDims{
        .batch = batch_size,
        .in_channels = in_channels,
        .in_height = in_height,
        .in_width = in_width,
        .out_channels = out_channels,
        .out_height = out_height,
        .out_width = out_width,
        .kernel_height = kernel_height,
        .kernel_width = kernel_width,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = pad_h_begin,
        .pad_w = pad_w_begin,
        .dilation_h = dilation_h,
        .dilation_w = dilation_w,
        .groups = actual_group,
        .group_in_channels = in_channels / actual_group,
        .group_out_channels = out_channels / actual_group,
    };

    const layout = ConvLayout{
        .input_batch_stride = in_channels * in_height * in_width,
        .input_channel_stride = in_height * in_width,
        .input_row_stride = in_width,
        .weight_out_stride = weight_in_channels * kernel_height * kernel_width,
        .weight_channel_stride = kernel_height * kernel_width,
        .weight_row_stride = kernel_width,
        .output_batch_stride = out_channels * out_height * out_width,
        .output_channel_stride = out_height * out_width,
        .output_row_stride = out_width,
    };

    var channel_params = try pkg_allocator.alloc(ChannelParams, out_channels);
    defer pkg_allocator.free(channel_params);

    const bias_tensor = if (bias) |b| b else null;
    const bias_is_int = isInt(BiasType);

    for (0..out_channels) |m| {
        const w_scale_val: f32 = if (w_scale.data.len == out_channels)
            asF32(ScaleType, w_scale.data[m])
        else
            asF32(ScaleType, w_scale.data[0]);

        const weight_scale_q16 = @as(i64, q16(w_scale_val));
        const combined_scale_q16 = saturateI128ToI64((@as(i128, x_scale_q16) * @as(i128, weight_scale_q16)) >> SCALE_SHIFT);
        const weight_zero_point = if (@typeInfo(@TypeOf(w_zero_point)) == .pointer and w_zero_point.data.len == 0)
            0
        else
            readPerChannelZP(w_zero_point, m, out_channels);

        const bias_q16 = if (bias_tensor) |b_tensor| blk: {
            if (b_tensor.data.len == 0) break :blk 0;
            const raw = if (b_tensor.data.len == 1) b_tensor.data[0] else b_tensor.data[m];
            var bias_real = asF32(BiasType, raw);
            if (bias_is_int) {
                bias_real *= x_scale_val * w_scale_val;
            }
            break :blk @as(i64, @intFromFloat(@round(bias_real * scale_factor)));
        } else 0;

        channel_params[m] = .{
            .combined_scale_q16 = combined_scale_q16,
            .weight_zero_point = weight_zero_point,
            .bias_q16 = bias_q16,
        };
    }

    // DEBUG: kernel selection
    // std.debug.print("QLINEAR_DEBUG: kernel={}x{} dilation={}x{}\n", .{kernel_height, kernel_width, dilation_h, dilation_w});
    if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
        // std.debug.print("QLINEAR_DEBUG: using conv3x3EmbeddedOptimized\n", .{});
        conv3x3EmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    } else if (kernel_height == 1 and kernel_width == 1) {
        // std.debug.print("QLINEAR_DEBUG: using conv1x1EmbeddedOptimized\n", .{});
        conv1x1EmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    } else {
        // std.debug.print("QLINEAR_DEBUG: using convGenericEmbeddedOptimized\n", .{});
        convGenericEmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    }
}

inline fn conv3x3EmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const input_zp = quant.input_zero_point;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc_raw: i64 = 0;

                        for (0..dims.group_in_channels) |ic| {
                            const c = in_group_base + ic;
                            const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                            const weight_channel_base = weight_base + ic * layout.weight_channel_stride;

                            var kh: usize = 0;
                            while (kh < 3) : (kh += 1) {
                                const ih = ih_origin + @as(isize, @intCast(kh));
                                if (ih < 0 or ih >= in_height_isize) continue;

                                const input_row_base = input_channel_base + @as(usize, @intCast(ih)) * layout.input_row_stride;
                                const weight_row_base = weight_channel_base + kh * layout.weight_row_stride;

                                var kw: usize = 0;
                                var weight_index = weight_row_base;
                                while (kw < 3) : (kw += 1) {
                                    const iw = iw_origin + @as(isize, @intCast(kw));
                                    if (iw >= 0 and iw < in_width_isize) {
                                        const input_index = input_row_base + @as(usize, @intCast(iw));
                                        const x_q = @as(i32, @intCast(x_data[input_index]));
                                        const w_q = @as(i32, @intCast(w_data[weight_index]));
                                        const x_diff = x_q - input_zp;
                                        const w_diff = w_q - channel.weight_zero_point;
                                        acc_raw += @as(i64, x_diff) * @as(i64, w_diff);
                                    }
                                    weight_index += 1;
                                }
                            }
                        }

                        const scaled = saturateI128ToI64(@as(i128, acc_raw) * @as(i128, channel.combined_scale_q16));
                        const sum = @addWithOverflow(channel.bias_q16, scaled);
                        var acc_q16: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_q16 = if (scaled >= 0) std.math.maxInt(i64) else std.math.minInt(i64);
                        }
                        const q = quantizeAccumulator(acc_q16, quant);
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

inline fn conv1x1EmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const input_zp = quant.input_zero_point;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc_raw: i64 = 0;

                        if (ih_origin >= 0 and ih_origin < in_height_isize and iw_origin >= 0 and iw_origin < in_width_isize) {
                            const ih = @as(usize, @intCast(ih_origin));
                            const iw = @as(usize, @intCast(iw_origin));

                            for (0..dims.group_in_channels) |ic| {
                                const c = in_group_base + ic;
                                const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                                const input_index = input_channel_base + ih * layout.input_row_stride + iw;
                                const weight_index = weight_base + ic;

                                const x_q = @as(i32, @intCast(x_data[input_index]));
                                const w_q = @as(i32, @intCast(w_data[weight_index]));
                                const x_diff = x_q - input_zp;
                                const w_diff = w_q - channel.weight_zero_point;
                                acc_raw += @as(i64, x_diff) * @as(i64, w_diff);
                            }
                        }

                        const scaled = saturateI128ToI64(@as(i128, acc_raw) * @as(i128, channel.combined_scale_q16));
                        const sum = @addWithOverflow(channel.bias_q16, scaled);
                        var acc_q16: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_q16 = if (scaled >= 0) std.math.maxInt(i64) else std.math.minInt(i64);
                        }
                        const q = quantizeAccumulator(acc_q16, quant);
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

inline fn convGenericEmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const input_zp = quant.input_zero_point;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc_raw: i64 = 0;

                        for (0..dims.group_in_channels) |ic| {
                            const c = in_group_base + ic;
                            const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                            const weight_channel_base = weight_base + ic * layout.weight_channel_stride;

                            var kh: usize = 0;
                            while (kh < dims.kernel_height) : (kh += 1) {
                                const ih = ih_origin + @as(isize, @intCast(kh * dims.dilation_h));
                                if (ih < 0 or ih >= in_height_isize) continue;

                                const input_row_base = input_channel_base + @as(usize, @intCast(ih)) * layout.input_row_stride;
                                const weight_row_base = weight_channel_base + kh * layout.weight_row_stride;

                                var kw: usize = 0;
                                while (kw < dims.kernel_width) : (kw += 1) {
                                    const iw = iw_origin + @as(isize, @intCast(kw * dims.dilation_w));
                                    if (iw < 0 or iw >= in_width_isize) {
                                        continue;
                                    }

                                    const input_index = input_row_base + @as(usize, @intCast(iw));
                                    const weight_index = weight_row_base + kw;

                                    const x_q = @as(i32, @intCast(x_data[input_index]));
                                    const w_q = @as(i32, @intCast(w_data[weight_index]));
                                    const x_diff = x_q - input_zp;
                                    const w_diff = w_q - channel.weight_zero_point;
                                    acc_raw += @as(i64, x_diff) * @as(i64, w_diff);
                                }
                            }
                        }

                        const scaled = saturateI128ToI64(@as(i128, acc_raw) * @as(i128, channel.combined_scale_q16));
                        const sum = @addWithOverflow(channel.bias_q16, scaled);
                        var acc_q16: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_q16 = if (scaled >= 0) std.math.maxInt(i64) else std.math.minInt(i64);
                        }
                        const q = quantizeAccumulator(acc_q16, quant);

                        // Fixed point quantization now matches ONNX Runtime behavior

                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

// Helper function to quantize multiplier for CMSIS-NN
fn quantizeMultiplier(scale: f32, multiplier: *i32, shift: *i32) void {
    if (scale == 0.0) {
        multiplier.* = 0;
        shift.* = 0;
        return;
    }

    var sig = scale;
    var exp: i32 = 0;

    // Normalize to [0.5, 1.0) range
    while (sig >= 1.0) {
        sig /= 2.0;
        exp += 1;
    }
    while (sig < 0.5) {
        sig *= 2.0;
        exp -= 1;
    }

    // Convert to fixed point representation
    const fixed_point_multiplier = @as(i32, @intFromFloat(@round(sig * (1 << 31))));

    multiplier.* = fixed_point_multiplier;
    shift.* = exp;
}

/// Direct CMSIS-NN wrapper - passes quantized data directly with minimal overhead
/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
/// Compile-time dispatch function that chooses the best implementation
pub fn qlinearconv_dispatch(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime OutputType: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype,
    output: *Tensor(InputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    const accelerators = @import("../Accelerators/mod.zig");
    if (!accelerators.canUseCmsisHelium()) {
        // Reference build: force embedded fixed-point implementation
        // DEBUG: dispatch to embedded
        // std.debug.print("QLINEAR_DEBUG: dispatch -> qlinearconv_embedded_lean, input[0]={}\n", .{x.data[0]});
        return qlinearconv_embedded_lean(
            InputType,
            WeightType,
            ScaleType,
            void,
            BiasType,
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            output,
            y_scale,
            y_zero_point,
            bias,
            stride,
            pads,
            dilations,
            group,
            auto_pad,
        );
    }

    // CMSIS path
    return qlinearconv_cmsis_accelerated(
        InputType,
        WeightType,
        ScaleType,
        OutputType,
        BiasType,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        output,
        y_scale,
        y_zero_point,
        bias,
        stride,
        pads,
        dilations,
        group,
        auto_pad,
    );
}

/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
pub fn qlinearconv_cmsis_accelerated(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    _x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    _w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
    output: *Tensor(InputType),
    _y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    // Mark CMSIS usage for testing
    const accelerators = @import("../Accelerators/mod.zig");
    accelerators.markCmsisUsed();

    // DEBUG: Print function entry
    // std.debug.print("CMSIS DEBUG: qlinearconv_cmsis_accelerated called\n", .{});

    // Suppress unused parameter warnings

    // Basic validation
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");

    // Guard: CMSIS-NN conv s8 path (generic and wrappers) does not support dilation != 1 on many kernels.
    // Fallback to embedded reference when dilation is used to avoid runtime errors (rc=-1).
    if (dilation_h != 1 or dilation_w != 1) {
        return qlinearconv_embedded_lean(
            InputType,
            WeightType,
            ScaleType,
            void,
            BiasType,
            x,
            _x_scale,
            x_zero_point,
            w,
            _w_scale,
            w_zero_point_any,
            output,
            _y_scale,
            y_zero_point,
            bias,
            stride,
            pads,
            dilations,
            group,
            "NOTSET",
        );
    }

    // Extract dimensions
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    var in_height = x.shape[2];
    var in_width = x.shape[3];
    const out_channels = output.shape[1];
    const weight_dim0 = w.shape[0];
    const kernel_height = w.shape[1];
    const kernel_width = w.shape[2];
    const weight_last_dim = w.shape[3];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    var pad_h = pads_arr[0];
    var pad_w = pads_arr[1];
    var pad_bottom: usize = pad_h;
    var pad_right: usize = pad_w;
    if (pads) |p| {
        if (p.len >= 4) {
            pad_bottom = p[2];
            pad_right = p[3];
        }
    }

    const input_zero_point = readScalarZP(InputType, x_zero_point);
    const output_zero_point = readScalarZP(InputType, y_zero_point);

    var padded_input: ?Tensor(InputType) = null;
    defer if (padded_input) |*tensor| tensor.deinit();
    if (pad_bottom != pad_h or pad_right != pad_w) {
        const padded = try applyAsymmetricPadding(InputType, x, pad_h, pad_w, pad_bottom, pad_right, input_zero_point);
        padded_input = padded;
        const padded_ref = &padded_input.?;
        in_height = padded_ref.shape[2];
        in_width = padded_ref.shape[3];
        pad_h = 0;
        pad_w = 0;
    }

    const effective_input: *const Tensor(InputType) = if (padded_input) |*tensor| tensor else x;

    if (group_val == 0) {
        return TensorMathError.InvalidDimensions;
    }
    const has_depthwise_layout = weight_dim0 == 1 and weight_last_dim == out_channels;
    const has_legacy_depthwise_layout =
        group_val == in_channels and weight_dim0 == out_channels and weight_last_dim == 1;
    const is_depthwise = group_val == in_channels and (has_depthwise_layout or has_legacy_depthwise_layout);
    const weight_in_channels = if (is_depthwise) 1 else weight_last_dim;

    if (is_depthwise) {
        // Depthwise layout already validated by has_depthwise_layout.
    } else {
        if (weight_dim0 != out_channels) {
            return TensorMathError.InvalidDimensions;
        }
    }

    if (weight_in_channels * group_val != in_channels or out_channels % group_val != 0) {
        return TensorMathError.InvalidDimensions;
    }

    const group_in_channels = weight_in_channels;
    const group_out_channels = out_channels / group_val;

    const expected_weights: usize = out_channels * group_in_channels * kernel_height * kernel_width;
    if (w.data.len != expected_weights) {
        return TensorMathError.InvalidDimensions;
    }

    if (@sizeOf(WeightType) != 1) {
        return TensorMathError.InvalidDataType;
    }

    // DEBUG: Print tensor dimensions
    // std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_channels, in_height, in_width });
    // std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, in_channels, kernel_height, kernel_width });
    // std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_channels, out_height, out_width });
    // std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // DEBUG: Print zero points
    // std.debug.print("CMSIS DEBUG: input_zero_point: {}, output_zero_point: {}\n", .{ input_zero_point, output_zero_point });

    // Helper functions for zero point and scale extraction
    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    // Extract quantization parameters
    const x_scale_val = asF32(ScaleType, _x_scale.data[0]);
    const y_scale_val = asF32(ScaleType, _y_scale.data[0]);
    const w_scale_data = _w_scale.data;
    const has_per_channel_w_scale = w_scale_data.len == out_channels;

    // CMSIS expects shifts within [-31, 31] and multiplier in Q31. We'll compute robustly.
    const multipliers_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(multipliers_buf);
    const shifts_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(shifts_buf);

    for (0..out_channels) |ch| {
        const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
        var scale_ratio = (x_scale_val * w_scale_val) / y_scale_val;
        if (!(scale_ratio > 0)) scale_ratio = 1.0; // guard against zero/neg due to malformed quant params
        quantizeMultiplier(scale_ratio, &multipliers_buf[ch], &shifts_buf[ch]);
    }

    // Now implementing the actual CMSIS-NN convolution with proper u8 to i8 conversion

    // Setup CMSIS-NN dimensions
    // CMSIS expects dims as int32. Guard against overflow for safety on large tensors
    const max_i32_u64: u64 = 2147483647;
    if (@as(u64, batch_size) > max_i32_u64 or
        @as(u64, in_height) > max_i32_u64 or
        @as(u64, in_width) > max_i32_u64 or
        @as(u64, in_channels) > max_i32_u64 or
        @as(u64, out_height) > max_i32_u64 or
        @as(u64, out_width) > max_i32_u64 or
        @as(u64, out_channels) > max_i32_u64)
    {
        return TensorMathError.InvalidDimensions;
    }
    var input_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(in_channels) };
    var output_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(out_channels) };
    var bias_dims = cmsis_nn.Dims{ .n = 1, .h = 1, .w = 1, .c = @intCast(out_channels) };
    var input_group_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(group_in_channels) };
    var output_group_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(group_out_channels) };
    var bias_group_dims = cmsis_nn.Dims{ .n = 1, .h = 1, .w = 1, .c = @intCast(group_out_channels) };

    // Proper CMSIS-NN offset calculation and data conversion
    // CMSIS arm_convolve_s8 expects s8 input/output and uses offsets in the same s8 domain.
    // If our tensors are u8, convert data to s8 domain by subtracting 128, and convert offsets accordingly.
    const is_u8_input = InputType == u8;
    const input_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(input_zero_point)) - 128
    else
        @as(i32, @intCast(input_zero_point));
    const output_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(output_zero_point)) - 128
    else
        @as(i32, @intCast(output_zero_point));

    // CMSIS s8 uses: input_offset = -x_zp_s8, output_offset = y_zp_s8
    // Clamp to i32 to avoid UB with large zero-points
    const cmsis_input_offset: i32 = -@as(i32, @intCast(input_zero_point_s8));
    const cmsis_output_offset: i32 = @as(i32, @intCast(output_zero_point_s8));

    var conv_params = cmsis_nn.ConvParams{
        .input_offset = cmsis_input_offset,
        .output_offset = cmsis_output_offset,
        .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
        .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
        .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
        // CMSIS s8 kernels clamp in s8 domain
        .activation = .{ .min = -128, .max = 127 },
    };

    var quant_params = cmsis_nn.PerChannelQuantParams{
        .multiplier = multipliers_buf.ptr,
        .shift = shifts_buf.ptr,
    };

    // Convert bias to i32 format as expected by CMSIS-NN
    var bias_converted: ?[]i32 = null;
    defer if (bias_converted) |buf| pkg_allocator.free(buf);
    var bias_ptr: ?[*]const i32 = null;

    if (bias) |b| {
        const bias_buf = try pkg_allocator.alloc(i32, out_channels);
        const has_per_channel_bias = b.data.len == out_channels;
        const bias_min = std.math.minInt(i32);
        const bias_max = std.math.maxInt(i32);
        var bias_slice = bias_buf;
        for (0..out_channels) |ch| {
            const bias_val = if (has_per_channel_bias) b.data[ch] else b.data[0];
            switch (@typeInfo(BiasType)) {
                .int, .comptime_int => {
                    bias_slice[ch] = @as(i32, @intCast(bias_val));
                },
                .float => {
                    const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
                    const bias_scale = x_scale_val * w_scale_val;
                    if (!(bias_scale > 0)) {
                        bias_slice[ch] = 0;
                        continue;
                    }
                    const bias_float = asF32(BiasType, bias_val);
                    const scaled = bias_float / bias_scale;
                    const clamped = std.math.clamp(scaled, @as(f32, @floatFromInt(bias_min)), @as(f32, @floatFromInt(bias_max)));
                    bias_slice[ch] = @as(i32, @intFromFloat(@round(clamped)));
                },
                else => @compileError("Unsupported bias tensor type for CMSIS-NN qlinearconv"),
            }
        }
        bias_converted = bias_slice;
        bias_ptr = @ptrCast(bias_slice.ptr);
    }

    var depthwise_repacked: ?[]i8 = null;
    defer if (depthwise_repacked) |buf| pkg_allocator.free(buf);

    const weight_ptr_s8: [*]const i8 = blk: {
        if (is_depthwise and !has_depthwise_layout) {
            const weights_per_channel = kernel_height * kernel_width;
            const total_weights = out_channels * weights_per_channel;
            const repacked = try pkg_allocator.alloc(i8, total_weights);
            depthwise_repacked = repacked;

            var oc: usize = 0;
            while (oc < out_channels) : (oc += 1) {
                var kh: usize = 0;
                while (kh < kernel_height) : (kh += 1) {
                    var kw: usize = 0;
                    while (kw < kernel_width) : (kw += 1) {
                        const src_index = (((oc * kernel_height) + kh) * kernel_width + kw) * weight_last_dim;
                        const dst_index = (((0 * kernel_height) + kh) * kernel_width + kw) * out_channels + oc;
                        repacked[dst_index] = @bitCast(i8, w.data[src_index]);
                    }
                }
            }

            break :blk repacked.ptr;
        }

        break :blk @ptrCast(w.data.ptr);
    };

    // Allocate buffer required by CMSIS-NN wrapper (regular or depthwise)
    var filter_dims = cmsis_nn.Dims{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
    var filter_group_dims = cmsis_nn.Dims{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
    var buffer_size: i32 = 0;
    if (is_depthwise) {
        buffer_size = cmsis_nn.conv.arm_depthwise_conv_wrapper_s8_get_buffer_size(&.{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        }, &input_dims, &.{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) }, &output_dims);
    } else if (group_val == 1) {
        buffer_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    } else {
        buffer_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_group_dims, &filter_group_dims, &output_group_dims);
    }
    if (buffer_size < 0) buffer_size = 0;
    var dyn_buffer: ?[]u8 = null;
    defer if (dyn_buffer) |buf| pkg_allocator.free(buf);
    var buffer_ptr: ?*anyopaque = null;
    var ctx_size_i32: i32 = 0;
    if (buffer_size > 0) {
        // CMSIS-NN scratch typically needs 4-byte alignment
        const buf = try pkg_allocator.alignedAlloc(u8, @alignOf(i32), @intCast(buffer_size));
        dyn_buffer = buf;
        buffer_ptr = buf.ptr;
        ctx_size_i32 = @intCast(buf.len);
    }

    var ctx = cmsis_nn.Context{ .buf = buffer_ptr, .size = ctx_size_i32 };
    // Wrapper API does not use upscale_dims

    // Prepare input/output pointers in s8 NHWC domain
    var input_converted: ?[]i8 = null;
    var output_converted: ?[]i8 = null;
    var grouped_input: ?[]i8 = null;
    var grouped_output: ?[]i8 = null;
    defer if (input_converted) |buf| pkg_allocator.free(buf);
    defer if (output_converted) |buf| pkg_allocator.free(buf);
    defer if (grouped_input) |buf| pkg_allocator.free(buf);
    defer if (grouped_output) |buf| pkg_allocator.free(buf);

    // Convert input from NCHW (our layout) to NHWC (CMSIS layout) and to s8
    const input_len = effective_input.data.len;
    const input_buf = try pkg_allocator.alloc(i8, input_len);
    input_converted = input_buf;
    {
        const spatial_size = in_height * in_width;
        const batch_stride = spatial_size * in_channels;
        const zero_adjust: i32 = if (is_u8_input) 128 else 0;
        var n: usize = 0;
        var src_batch_base: usize = 0;
        var dst_batch_base: usize = 0;
        while (n < batch_size) : (n += 1) {
            var pixel: usize = 0;
            while (pixel < spatial_size) : (pixel += 1) {
                var src_idx = src_batch_base + pixel;
                var dst_idx = dst_batch_base + pixel * in_channels;
                var c: usize = 0;
                while (c < in_channels) : (c += 1) {
                    const raw = @as(i32, @intCast(effective_input.data[src_idx]));
                    const centered = raw - zero_adjust;
                    input_buf[dst_idx] = @as(i8, @intCast(std.math.clamp(centered, -128, 127)));
                    src_idx += spatial_size;
                    dst_idx += 1;
                }
            }
            src_batch_base += batch_stride;
            dst_batch_base += batch_stride;
        }
    }
    const input_ptr_s8: [*]const i8 = input_buf.ptr;

    // Always use a temporary NHWC s8 buffer for output (convert+reorder back after)
    const output_len = output.data.len;
    const output_buf = try pkg_allocator.alloc(i8, output_len);
    output_converted = output_buf;
    const output_ptr_s8: [*]i8 = output_buf.ptr;

    if (group_val != 1 and !is_depthwise) {
        const grouped_input_len = batch_size * in_height * in_width * group_in_channels;
        const grouped_output_len = batch_size * out_height * out_width * group_out_channels;
        grouped_input = try pkg_allocator.alloc(i8, grouped_input_len);
        grouped_output = try pkg_allocator.alloc(i8, grouped_output_len);
    }

    // Call CMSIS-NN wrapper (regular or depthwise)
    var status = cmsis_nn.ARM_CMSIS_NN_SUCCESS;
    if (is_depthwise) {
        var dw_params = cmsis_nn.DwConvParams{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        };
        // Depthwise expects filter dims [1, kh, kw, C_out]
        var dw_filter_dims = cmsis_nn.Dims{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) };
        status = cmsis_nn.conv.arm_depthwise_conv_wrapper_s8(
            &ctx,
            &dw_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &dw_filter_dims,
            weight_ptr_s8,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else if (group_val == 1) {
        status = cmsis_nn.conv.arm_convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &filter_dims,
            weight_ptr_s8,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
        if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
            // Try direct (non-wrapper) kernel as a fallback while still using CMSIS
            var direct_buf_size = cmsis_nn.conv.arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
            if (direct_buf_size < 0) direct_buf_size = 0;
            var direct_dyn: ?[]u8 = null;
            defer if (direct_dyn) |b| pkg_allocator.free(b);
            var direct_ctx = ctx;
            if (direct_buf_size > ctx_size_i32) {
                const newb = try pkg_allocator.alignedAlloc(u8, @alignOf(i32), @intCast(direct_buf_size));
                direct_dyn = newb;
                direct_ctx = .{ .buf = newb.ptr, .size = @intCast(newb.len) };
            }
            status = cmsis_nn.conv.arm_convolve_s8(
                &direct_ctx,
                &conv_params,
                &quant_params,
                &input_dims,
                input_ptr_s8,
                &filter_dims,
                weight_ptr_s8,
                &bias_dims,
                if (bias_ptr) |ptr| @ptrCast(ptr) else null,
                null,
                &output_dims,
                output_ptr_s8,
            );
        }
    } else {
        const grouped_in_buf = grouped_input.?;
        const grouped_out_buf = grouped_output.?;
        const grouped_in_ptr: [*]const i8 = grouped_in_buf.ptr;
        const grouped_out_ptr: [*]i8 = grouped_out_buf.ptr;
        const total_input_pixels_group = batch_size * in_height * in_width;
        const total_output_pixels_group = batch_size * out_height * out_width;
        var g: usize = 0;
        while (g < group_val) : (g += 1) {
            const channel_offset_in = g * group_in_channels;
            const channel_offset_out = g * group_out_channels;
            for (0..total_input_pixels_group) |idx| {
                const src_base = idx * in_channels + channel_offset_in;
                const dst_base = idx * group_in_channels;
                std.mem.copyForwards(i8, grouped_in_buf[dst_base .. dst_base + group_in_channels], input_buf[src_base .. src_base + group_in_channels]);
            }

            const group_channel_offset = g * group_out_channels;
            var group_quant_params = cmsis_nn.PerChannelQuantParams{
                .multiplier = multipliers_buf[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
                .shift = shifts_buf[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
            };
            const weights_offset = g * group_out_channels * group_in_channels * kernel_height * kernel_width;
            const group_weights_ptr = weight_ptr_s8 + weights_offset;
            const bias_group_ptr = if (bias_ptr) |ptr| ptr + group_channel_offset else null;

            status = cmsis_nn.conv.arm_convolve_wrapper_s8(
                &ctx,
                &conv_params,
                &group_quant_params,
                &input_group_dims,
                grouped_in_ptr,
                &filter_group_dims,
                group_weights_ptr,
                &bias_group_dims,
                if (bias_group_ptr) |ptr| @ptrCast(ptr) else null,
                &output_group_dims,
                grouped_out_ptr,
            );
            if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
                // Try direct group call
                var direct_buf_size_g = cmsis_nn.conv.arm_convolve_s8_get_buffer_size(&input_group_dims, &filter_group_dims);
                if (direct_buf_size_g < 0) direct_buf_size_g = 0;
                var direct_dyn_g: ?[]u8 = null;
                defer if (direct_dyn_g) |b| pkg_allocator.free(b);
                var direct_ctx_g = ctx;
                if (direct_buf_size_g > ctx_size_i32) {
                    const newbg = try pkg_allocator.alignedAlloc(u8, @alignOf(i32), @intCast(direct_buf_size_g));
                    direct_dyn_g = newbg;
                    direct_ctx_g = .{ .buf = newbg.ptr, .size = @intCast(newbg.len) };
                }
                status = cmsis_nn.conv.arm_convolve_s8(
                    &direct_ctx_g,
                    &conv_params,
                    &group_quant_params,
                    &input_group_dims,
                    grouped_in_ptr,
                    &filter_group_dims,
                    group_weights_ptr,
                    &bias_group_dims,
                    if (bias_group_ptr) |ptr| @ptrCast(ptr) else null,
                    null,
                    &output_group_dims,
                    grouped_out_ptr,
                );
                if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) break;
            }

            for (0..total_output_pixels_group) |idx| {
                const src_base = idx * group_out_channels;
                const dst_base = idx * out_channels + channel_offset_out;
                std.mem.copyForwards(i8, output_buf[dst_base .. dst_base + group_out_channels], grouped_out_buf[src_base .. src_base + group_out_channels]);
            }
        }
    }

    if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
        // Fallback to embedded implementation on unsupported CMSIS-NN configurations
        return qlinearconv_embedded_lean(
            InputType,
            WeightType,
            ScaleType,
            void,
            BiasType,
            x,
            _x_scale,
            x_zero_point,
            w,
            _w_scale,
            w_zero_point_any,
            output,
            _y_scale,
            y_zero_point,
            bias,
            stride,
            pads,
            dilations,
            group,
            auto_pad,
        );
    }

    // Reorder output from NHWC back to NCHW and convert s8 -> u8 if needed
    {
        const buf = output_converted.?;
        const spatial_size = out_height * out_width;
        const batch_stride = spatial_size * out_channels;
        const zero_restore: i32 = if (is_u8_input) 128 else 0;
        var n: usize = 0;
        var src_batch_base: usize = 0;
        var dst_batch_base: usize = 0;
        while (n < batch_size) : (n += 1) {
            var pixel: usize = 0;
            while (pixel < spatial_size) : (pixel += 1) {
                var src_idx = src_batch_base + pixel * out_channels;
                var dst_idx = dst_batch_base + pixel;
                var c: usize = 0;
                while (c < out_channels) : (c += 1) {
                    const v_i32 = @as(i32, @intCast(buf[src_idx]));
                    const adjusted = v_i32 + zero_restore;
                    if (is_u8_input) {
                        output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
                    } else {
                        output.data[dst_idx] = @as(InputType, @intCast(adjusted));
                    }
                    src_idx += 1;
                    dst_idx += spatial_size;
                }
            }
            src_batch_base += batch_stride;
            dst_batch_base += batch_stride;
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
