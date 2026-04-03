const std = @import("std");
const zant = @import("../../../zant.zig");

const SCALE_SHIFT: u5 = 16;

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("../op_conv/zant_conv.zig");

// Import the main qlinearconv file for the primary lean function (used as fallback)
const main_qlinearconv = @import("zant_qlinearconv.zig");

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
    return @as(f32, @floatCast(s.data[0]));
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

inline fn saturateToI64(value: i128) i64 {
    if (value > @as(i128, std.math.maxInt(i64))) return std.math.maxInt(i64);
    if (value < @as(i128, std.math.minInt(i64))) return std.math.minInt(i64);
    return @as(i64, @intCast(value));
}

inline fn saturatingShiftLeft(value: i64, shift: u6) i64 {
    if (shift >= 63) {
        return if (value >= 0) std.math.maxInt(i64) else std.math.minInt(i64);
    }

    const shifted = @shlWithOverflow(value, shift);
    if (shifted[1] != 0) {
        return if (value >= 0) std.math.maxInt(i64) else std.math.minInt(i64);
    }
    return shifted[0];
}

inline fn clampToI8(v: anytype) i8 {
    const val_i32: i32 = @as(i32, @intCast(v));
    const clamped = std.math.clamp(val_i32, std.math.minInt(i8), std.math.maxInt(i8));
    return @as(i8, @intCast(clamped));
}

inline fn roundingDivideByPOT(value: i64, exponent: u6) i64 {
    if (exponent == 0) return value;

    const denom: i64 = @as(i64, 1) << exponent;
    const mask: i64 = denom - 1;
    var result = value >> exponent;
    const remainder = value & mask;

    if (value < 0 and remainder != 0) {
        result += 1;
    }

    const abs_rem = if (value >= 0)
        remainder
    else
        (denom - remainder) & mask;
    const half: i64 = denom >> 1;
    const sign: i64 = if (value >= 0) 1 else -1;

    if (abs_rem > half) {
        result += sign;
    } else if (abs_rem == half) {
        if ((result & 1) != 0) result += sign;
    }

    return result;
}

inline fn requantize(
    value: i64,
    multiplier: i32,
    shift: i32,
    q_min: i32,
    q_max: i32,
    output_zero_point: i32,
) i32 {
    if (multiplier == 0) {
        return std.math.clamp(output_zero_point, q_min, q_max);
    }

    var product = @as(i128, value) * @as(i128, multiplier);
    const rounding: i128 = if (product >= 0) (@as(i128, 1) << 30) else -(@as(i128, 1) << 30);
    product += rounding;
    product >>= 31;

    var scaled = saturateToI64(product);

    if (shift > 0) {
        scaled = saturatingShiftLeft(scaled, @as(u6, @intCast(shift)));
    } else if (shift < 0) {
        scaled = roundingDivideByPOT(scaled, @as(u6, @intCast(-shift)));
    }

    const with_zp_128 = @as(i128, scaled) + @as(i128, output_zero_point);
    const with_zp = saturateToI64(with_zp_128);
    const clamped = std.math.clamp(with_zp, @as(i64, q_min), @as(i64, q_max));
    return @as(i32, @intCast(clamped));
}

const ChannelParams = struct {
    requant_multiplier: i32,
    requant_shift: i32,
    weight_zero_point: i32,
    bias_acc: i64,
};

const QuantParams = struct {
    input_zero_point: i32,
    output_zero_point_q16: i64,
    q_min: i32,
    q_max: i32,
};

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
    coreLogStatic("QLINEAR: using qlinearconv_embedded_lean (int)\n");
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
        return main_qlinearconv.qlinear_conv_lean(InputType, WeightType, ScaleType, void, BiasType, x, x_scale, x_zero_point, w, w_scale, w_zero_point, output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad);
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

    // use module-level SCALE_SHIFT
    const x_scale_val = asF32(ScaleType, x_scale.data[0]);
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
        .input_zero_point = input_zero_point,
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

        var multiplier: i32 = 0;
        var shift: i32 = 0;
        const total_scale = if (y_scale_val == 0)
            0
        else
            (x_scale_val * w_scale_val) / y_scale_val;
        quantizeMultiplier(total_scale, &multiplier, &shift);
        const weight_zero_point = if (@typeInfo(@TypeOf(w_zero_point)) == .pointer and w_zero_point.data.len == 0)
            0
        else
            readPerChannelZP(w_zero_point, m, out_channels);

        const bias_acc = if (bias_tensor) |b_tensor| blk: {
            if (b_tensor.data.len == 0) break :blk 0;
            const raw = if (b_tensor.data.len == 1) b_tensor.data[0] else b_tensor.data[m];
            var bias_real = asF32(BiasType, raw);
            if (bias_is_int) {
                bias_real *= x_scale_val * w_scale_val;
            }
            if (total_scale == 0) {
                if (y_scale_val == 0) break :blk 0;
                break :blk @as(i64, @intFromFloat(@round(bias_real / y_scale_val)));
            }
            const acc_scale = x_scale_val * w_scale_val;
            if (acc_scale == 0) break :blk 0;
            break :blk @as(i64, @intFromFloat(@round(bias_real / acc_scale)));
        } else 0;

        channel_params[m] = .{
            .requant_multiplier = multiplier,
            .requant_shift = shift,
            .weight_zero_point = weight_zero_point,
            .bias_acc = bias_acc,
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
                const use_simd = isEightBitInt(InputType) and isEightBitInt(WeightType);

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc_raw: i64 = 0;
                        var ic_start: usize = 0;

                        if (use_simd) {
                            const simd = dotProductKernelSimd(
                                InputType,
                                WeightType,
                                x_data,
                                w_data,
                                dims,
                                layout,
                                input_batch_base,
                                weight_base,
                                in_group_base,
                                ih_origin,
                                iw_origin,
                                3,
                                3,
                                1,
                                1,
                                input_zp,
                                channel,
                                in_height_isize,
                                in_width_isize,
                            );
                            logSimdEvent(
                                "QLINEAR_SIMD: conv3x3 n={} g={} oc={} oh={} ow={} simd_acc={} processed={} remainder={}\n",
                                .{
                                    n,
                                    g,
                                    m,
                                    oh,
                                    ow,
                                    simd.acc,
                                    simd.processed,
                                    dims.group_in_channels - simd.processed,
                                },
                            );
                            acc_raw += simd.acc;
                            ic_start = simd.processed;
                        }

                        for (ic_start..dims.group_in_channels) |ic| {
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

                        if (channel.requant_multiplier == 0) {
                            const base = @as(i64, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT));
                            const base_sum = @addWithOverflow(base, channel.bias_acc);
                            var biased_i64: i64 = base_sum[0];
                            if (base_sum[1] != 0) {
                                biased_i64 = if ((base >= 0 and channel.bias_acc >= 0))
                                    std.math.maxInt(i64)
                                else
                                    std.math.minInt(i64);
                            }
                            const biased = std.math.clamp(
                                biased_i64,
                                @as(i64, quant.q_min),
                                @as(i64, quant.q_max),
                            );
                            out_data[output_row_base + ow] = @as(InputType, @intCast(biased));
                            continue;
                        }

                        const sum = @addWithOverflow(acc_raw, channel.bias_acc);
                        var acc_with_bias: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_with_bias = if ((acc_raw >= 0 and channel.bias_acc >= 0))
                                std.math.maxInt(i64)
                            else
                                std.math.minInt(i64);
                        }
                        const q = requantize(
                            acc_with_bias,
                            channel.requant_multiplier,
                            channel.requant_shift,
                            quant.q_min,
                            quant.q_max,
                            @as(i32, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT)),
                        );
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

inline fn isEightBitInt(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => |info| info.bits == 8,
        else => false,
    };
}

const simd_block = 16;
const enable_simd_logging = false;

inline fn logSimdEvent(comptime fmt: []const u8, args: anytype) void {
    if (!enable_simd_logging) return;
    coreLogf(fmt, args);
}

inline fn dotProductKernelSimd(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    dims: ConvDims,
    layout: ConvLayout,
    input_batch_base: usize,
    weight_base: usize,
    in_group_base: usize,
    ih_origin: isize,
    iw_origin: isize,
    kernel_height: usize,
    kernel_width: usize,
    dilation_h: usize,
    dilation_w: usize,
    input_zp: i32,
    channel: ChannelParams,
    in_height_isize: isize,
    in_width_isize: isize,
) struct { acc: i64, processed: usize } {
    if (dims.group_in_channels < simd_block) {
        logSimdEvent(
            "QLINEAR_SIMD: skip SIMD because channels={} < block={}\n",
            .{ dims.group_in_channels, simd_block },
        );
        return .{ .acc = 0, .processed = 0 };
    }

    var acc_total: i64 = 0;
    var processed: usize = 0;
    const weight_zp_vec = @as(@Vector(simd_block, i32), @splat(channel.weight_zero_point));

    while (processed + simd_block <= dims.group_in_channels) : (processed += simd_block) {
        logSimdEvent(
            "QLINEAR_SIMD: processing block start={} end={} kernel={}x{} dil={}x{}\n",
            .{
                processed,
                processed + simd_block,
                kernel_height,
                kernel_width,
                dilation_h,
                dilation_w,
            },
        );
        var block_acc = @as(@Vector(simd_block, i64), @splat(0));

        var kh: usize = 0;
        while (kh < kernel_height) : (kh += 1) {
            const ih = ih_origin + @as(isize, @intCast(kh * dilation_h));
            if (ih < 0 or ih >= in_height_isize) continue;

            const ih_usize = @as(usize, @intCast(ih));

            var kw: usize = 0;
            while (kw < kernel_width) : (kw += 1) {
                const iw = iw_origin + @as(isize, @intCast(kw * dilation_w));
                if (iw < 0 or iw >= in_width_isize) continue;

                const iw_usize = @as(usize, @intCast(iw));
                var x_block: [simd_block]i32 = undefined;
                var w_block: [simd_block]i32 = undefined;

                inline for (0..simd_block) |lane| {
                    const channel_index = in_group_base + processed + lane;
                    const input_channel_base = input_batch_base + channel_index * layout.input_channel_stride;
                    const input_row_base = input_channel_base + ih_usize * layout.input_row_stride;
                    const input_index = input_row_base + iw_usize;
                    const x_q = @as(i32, @intCast(x_data[input_index]));
                    x_block[lane] = x_q - input_zp;

                    const lane_weight_channel_base = weight_base + (processed + lane) * layout.weight_channel_stride;
                    const lane_weight_row_base = lane_weight_channel_base + kh * layout.weight_row_stride;
                    const weight_index = lane_weight_row_base + kw;
                    w_block[lane] = @as(i32, @intCast(w_data[weight_index]));
                }

                const x_vec = @as(@Vector(simd_block, i32), x_block);
                const w_vec = @as(@Vector(simd_block, i32), w_block) - weight_zp_vec;
                const prod = @as(@Vector(simd_block, i64), @intCast(x_vec)) *
                    @as(@Vector(simd_block, i64), @intCast(w_vec));
                block_acc = block_acc + prod;
            }
        }

        const block_sum = @reduce(.Add, block_acc);
        acc_total += block_sum;
        logSimdEvent(
            "QLINEAR_SIMD: block complete processed={} acc_block={} acc_total={}\n",
            .{ processed + simd_block, block_sum, acc_total },
        );
    }

    logSimdEvent(
        "QLINEAR_SIMD: finished processed={} of {} channels acc={} remainder={}\n",
        .{ processed, dims.group_in_channels, acc_total, dims.group_in_channels - processed },
    );

    return .{ .acc = acc_total, .processed = processed };
}

inline fn dotProduct1x1Simd(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    dims: ConvDims,
    layout: ConvLayout,
    input_batch_base: usize,
    weight_base: usize,
    in_group_base: usize,
    ih: usize,
    iw: usize,
    input_zp: i32,
    channel: ChannelParams,
) i64 {
    const block: usize = 16;
    var acc: i64 = 0;
    const input_zp_vec = @as(@Vector(block, i32), @splat(input_zp));
    const weight_zp_vec = @as(@Vector(block, i32), @splat(channel.weight_zero_point));

    var processed: usize = 0;
    while (processed + block <= dims.group_in_channels) : (processed += block) {
        var input_block: [block]i32 = undefined;
        var weight_block: [block]i32 = undefined;

        inline for (0..block) |lane| {
            const channel_index = in_group_base + processed + lane;
            const input_channel_base = input_batch_base + channel_index * layout.input_channel_stride;
            const input_index = input_channel_base + ih * layout.input_row_stride + iw;
            input_block[lane] = @as(i32, @intCast(x_data[input_index]));
            weight_block[lane] = @as(i32, @intCast(w_data[weight_base + processed + lane]));
        }

        const x_vec = @as(@Vector(block, i32), input_block);
        const w_vec = @as(@Vector(block, i32), weight_block);
        const x_diff = x_vec - input_zp_vec;
        const w_diff = w_vec - weight_zp_vec;
        const products = @as(@Vector(block, i64), @intCast(x_diff)) * @as(@Vector(block, i64), @intCast(w_diff));
        acc += @reduce(.Add, products);
    }

    var remainder = processed;
    while (remainder < dims.group_in_channels) : (remainder += 1) {
        const c = in_group_base + remainder;
        const input_channel_base = input_batch_base + c * layout.input_channel_stride;
        const input_index = input_channel_base + ih * layout.input_row_stride + iw;
        const weight_index = weight_base + remainder;
        const x_q = @as(i32, @intCast(x_data[input_index]));
        const w_q = @as(i32, @intCast(w_data[weight_index]));
        const x_diff = x_q - input_zp;
        const w_diff = w_q - channel.weight_zero_point;
        acc += @as(i64, x_diff) * @as(i64, w_diff);
    }

    return acc;
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

                            if (comptime (isEightBitInt(InputType) and isEightBitInt(WeightType))) {
                                acc_raw = dotProduct1x1Simd(
                                    InputType,
                                    WeightType,
                                    x_data,
                                    w_data,
                                    dims,
                                    layout,
                                    input_batch_base,
                                    weight_base,
                                    in_group_base,
                                    ih,
                                    iw,
                                    input_zp,
                                    channel,
                                );
                            } else {
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
                        }

                        if (channel.requant_multiplier == 0) {
                            const base = @as(i64, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT));
                            const base_sum = @addWithOverflow(base, channel.bias_acc);
                            var biased_i64: i64 = base_sum[0];
                            if (base_sum[1] != 0) {
                                biased_i64 = if ((base >= 0 and channel.bias_acc >= 0))
                                    std.math.maxInt(i64)
                                else
                                    std.math.minInt(i64);
                            }
                            const biased = std.math.clamp(
                                biased_i64,
                                @as(i64, quant.q_min),
                                @as(i64, quant.q_max),
                            );
                            out_data[output_row_base + ow] = @as(InputType, @intCast(biased));
                            continue;
                        }

                        const sum = @addWithOverflow(acc_raw, channel.bias_acc);
                        var acc_with_bias: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_with_bias = if ((acc_raw >= 0 and channel.bias_acc >= 0))
                                std.math.maxInt(i64)
                            else
                                std.math.minInt(i64);
                        }
                        const q = requantize(
                            acc_with_bias,
                            channel.requant_multiplier,
                            channel.requant_shift,
                            quant.q_min,
                            quant.q_max,
                            @as(i32, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT)),
                        );
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
                const use_simd = isEightBitInt(InputType) and isEightBitInt(WeightType);

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc_raw: i64 = 0;
                        var ic_start: usize = 0;

                        if (use_simd) {
                            const simd = dotProductKernelSimd(
                                InputType,
                                WeightType,
                                x_data,
                                w_data,
                                dims,
                                layout,
                                input_batch_base,
                                weight_base,
                                in_group_base,
                                ih_origin,
                                iw_origin,
                                dims.kernel_height,
                                dims.kernel_width,
                                dims.dilation_h,
                                dims.dilation_w,
                                input_zp,
                                channel,
                                in_height_isize,
                                in_width_isize,
                            );
                            logSimdEvent(
                                "QLINEAR_SIMD: convGeneric n={} g={} oc={} oh={} ow={} simd_acc={} processed={} remainder={}\n",
                                .{
                                    n,
                                    g,
                                    m,
                                    oh,
                                    ow,
                                    simd.acc,
                                    simd.processed,
                                    dims.group_in_channels - simd.processed,
                                },
                            );
                            acc_raw += simd.acc;
                            ic_start = simd.processed;
                        }

                        for (ic_start..dims.group_in_channels) |ic| {
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

                        if (channel.requant_multiplier == 0) {
                            const base = @as(i64, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT));
                            const base_sum = @addWithOverflow(base, channel.bias_acc);
                            var biased_i64: i64 = base_sum[0];
                            if (base_sum[1] != 0) {
                                biased_i64 = if ((base >= 0 and channel.bias_acc >= 0))
                                    std.math.maxInt(i64)
                                else
                                    std.math.minInt(i64);
                            }
                            const biased = std.math.clamp(
                                biased_i64,
                                @as(i64, quant.q_min),
                                @as(i64, quant.q_max),
                            );
                            out_data[output_row_base + ow] = @as(InputType, @intCast(biased));
                            continue;
                        }

                        const sum = @addWithOverflow(acc_raw, channel.bias_acc);
                        var acc_with_bias: i64 = sum[0];
                        if (sum[1] != 0) {
                            acc_with_bias = if ((acc_raw >= 0 and channel.bias_acc >= 0))
                                std.math.maxInt(i64)
                            else
                                std.math.minInt(i64);
                        }
                        const q = requantize(
                            acc_with_bias,
                            channel.requant_multiplier,
                            channel.requant_shift,
                            quant.q_min,
                            quant.q_max,
                            @as(i32, @intCast(quant.output_zero_point_q16 >> SCALE_SHIFT)),
                        );

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
        coreLogStatic("QLINEAR: dispatch -> embedded_lean\n");
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
    coreLogStatic("QLINEAR: dispatch -> cmsis_accelerated\n");
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
    _: []const u8,
) !void {
    coreLogStatic("QLINEAR: using qlinearconv_cmsis_accelerated\n");
    // Mark CMSIS usage for testing
    const accelerators = @import("../Accelerators/mod.zig");
    accelerators.markCmsisUsed();

    // DEBUG: Print function entry
    // std.debug.print("CMSIS DEBUG: qlinearconv_cmsis_accelerated called\n", .{});

    // Suppress unused parameter warnings
    // w_zero_point_any is actually used later when packing weights (readPerChannelZP)

    // Basic validation
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");

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

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h = pads_arr[0];
    const pad_w = pads_arr[1];

    if (group_val == 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * group_val != in_channels or out_channels % group_val != 0) {
        return TensorMathError.InvalidDimensions;
    }

    const group_in_channels = weight_in_channels;
    const group_out_channels = out_channels / group_val;

    // DEBUG: Print tensor dimensions
    // std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_channels, in_height, in_width });
    // std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, in_channels, kernel_height, kernel_width });
    // std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_channels, out_height, out_width });
    // std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // Extract zero points
    const input_zero_point = readScalarZP(InputType, x_zero_point);
    const output_zero_point = readScalarZP(InputType, y_zero_point);

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

    const multipliers_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(multipliers_buf);
    const shifts_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(shifts_buf);

    for (0..out_channels) |ch| {
        const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
        const scale_ratio = (x_scale_val * w_scale_val) / y_scale_val;
        quantizeMultiplier(scale_ratio, &multipliers_buf[ch], &shifts_buf[ch]);
    }

    // Now implementing the actual CMSIS-NN convolution with proper u8 to i8 conversion

    // Setup CMSIS-NN dimensions
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

    const cmsis_input_offset = -input_zero_point_s8;
    const cmsis_output_offset = output_zero_point_s8;

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
        var bias_slice = bias_buf;
        for (0..out_channels) |ch| {
            const bias_val = if (has_per_channel_bias) b.data[ch] else b.data[0];
            const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
            const bias_scale = x_scale_val * w_scale_val;
            const bias_float = asF32(BiasType, bias_val);
            bias_slice[ch] = @as(i32, @intFromFloat(@round(bias_float / bias_scale)));
        }
        bias_converted = bias_slice;
        bias_ptr = @ptrCast(bias_slice.ptr);
    }

    // Pack weights depending on conv kind:
    // - Regular/grouped conv: OHWI (out, kh, kw, in)
    // - Depthwise conv: [1, kh, kw, C_out] per CMSIS DW wrapper
    const per_channel_w_zp = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(per_channel_w_zp);
    for (0..out_channels) |ch| {
        per_channel_w_zp[ch] = readPerChannelZP(w_zero_point_any, ch, out_channels);
    }

    const total_weights: usize = out_channels * weight_in_channels * kernel_height * kernel_width;
    var w_packed: []i8 = try pkg_allocator.alloc(i8, total_weights);
    defer pkg_allocator.free(w_packed);

    if (group_val == in_channels) {
        // Depthwise: expect C_out = in_channels * channel_multiplier and layout [1, kh, kw, C_out]
        const kernel_size = kernel_height * kernel_width;
        var wp: usize = 0;
        var spatial: usize = 0;
        while (spatial < kernel_size) : (spatial += 1) {
            var src_idx = spatial;
            var m: usize = 0;
            while (m < out_channels) : (m += 1) {
                const w_q_i32 = @as(i32, @intCast(w.data[src_idx]));
                w_packed[wp] = clampToI8(w_q_i32 - per_channel_w_zp[m]);
                src_idx += kernel_size; // next output channel's element at same spatial pos
                wp += 1;
            }
        }
    } else {
        // Regular and grouped convolution: pack weights in OHWI order (out, h, w, in)
        const kernel_size = kernel_height * kernel_width;
        const channel_stride = kernel_size;
        const output_stride = weight_in_channels * kernel_size;
        var wp: usize = 0;
        for (0..out_channels) |m| {
            const base_idx = m * output_stride;
            const w_zp = per_channel_w_zp[m];
            var spatial: usize = 0;
            while (spatial < kernel_size) : (spatial += 1) {
                var channel_idx = base_idx + spatial;
                var c: usize = 0;
                while (c < weight_in_channels) : (c += 1) {
                    const w_q_i32 = @as(i32, @intCast(w.data[channel_idx]));
                    w_packed[wp] = clampToI8(w_q_i32 - w_zp);
                    channel_idx += channel_stride;
                    wp += 1;
                }
            }
        }
    }

    // Allocate buffer required by CMSIS-NN wrapper (regular or depthwise)
    var filter_dims = cmsis_nn.Dims{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
    var filter_group_dims = cmsis_nn.Dims{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
    var buffer_size: i32 = 0;
    if (group_val == in_channels) {
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
        const buf = try pkg_allocator.alloc(u8, @intCast(buffer_size));
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
    const input_len = x.data.len;
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
                    const raw = @as(i32, @intCast(x.data[src_idx]));
                    input_buf[dst_idx] = @as(i8, @intCast(raw - zero_adjust));
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

    if (group_val != 1 and group_val != in_channels) {
        const grouped_input_len = batch_size * in_height * in_width * group_in_channels;
        const grouped_output_len = batch_size * out_height * out_width * group_out_channels;
        grouped_input = try pkg_allocator.alloc(i8, grouped_input_len);
        grouped_output = try pkg_allocator.alloc(i8, grouped_output_len);
    }

    // Call CMSIS-NN wrapper (regular or depthwise)
    var status = cmsis_nn.ARM_CMSIS_NN_SUCCESS;
    if (group_val == in_channels) {
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
            w_packed.ptr,
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
            w_packed.ptr,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
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
            const group_weights_ptr = w_packed.ptr + weights_offset;
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
                break;
            }

            for (0..total_output_pixels_group) |idx| {
                const src_base = idx * group_out_channels;
                const dst_base = idx * out_channels + channel_offset_out;
                std.mem.copyForwards(i8, output_buf[dst_base .. dst_base + group_out_channels], grouped_out_buf[src_base .. src_base + group_out_channels]);
            }
        }
    }

    if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
        return TensorMathError.UnexpectedError;
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
