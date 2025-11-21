const std = @import("std");
const zant = @import("../../../zant.zig");

const SCALE_SHIFT: u5 = 16;

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const mod_cmsis = @import("../Cmsis/mod_cmsis.zig");

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("op_convolution.zig");

// Import utilities for QLinearConv
const qlinearconvUtils = @import("tensor_math_utils/qlinearconv_utils.zig");

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
// Some of them have been moved to qlinearconv_utils.zig for reuse across multiple modules

inline fn readPerChannelScale(comptime T: type, s: *const Tensor(T), m: usize, M: usize) f32 {
    if (s.shape.len == 1 and s.shape[0] == M) return @as(f32, @floatCast(s.data[m]));
    return @as(f32, @floatCast(s.data[0])); // broadcast
}

inline fn selectChannelIndex(len: usize, channel: usize) usize {
    if (len <= 1) return 0;
    return if (channel < len) channel else len - 1;
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

    // Perform quantized convolution using dispatch to get the best implementation
    // use this to call cmsis or embedded version
    try qlinearconv_lean(InputType, WeightType, ScaleType, InputType, BiasType, input_ptr, x_scale, x_zero_point, w, w_scale, w_zero_point, &output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad.?);

    return output;
}

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
    const cmsis_enabled = comptime mod_cmsis.cmsisUsed();
    if (cmsis_enabled) {
        const cmsis_nn = @import("../Cmsis/wrappers/cmsis_nn.zig");
        return cmsis_nn.qlinearconv(
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
    } else {
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
        return qlinearconv_lean_fp(InputType, WeightType, ScaleType, void, BiasType, x, x_scale, x_zero_point, w, w_scale, w_zero_point, output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad);
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
        qlinearconvUtils.readScalarZP(InputType, x_zero_point);
    const output_zero_point = if (@typeInfo(@TypeOf(y_zero_point)) == .pointer and y_zero_point.data.len == 0)
        0
    else
        qlinearconvUtils.readScalarZP(InputType, y_zero_point);

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
        qlinearconvUtils.quantizeMultiplier(total_scale, &multiplier, &shift);
        const weight_zero_point = if (@typeInfo(@TypeOf(w_zero_point)) == .pointer and w_zero_point.data.len == 0)
            0
        else
            qlinearconvUtils.readPerChannelZP(w_zero_point, m, out_channels);

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

// ===== OPTIMIZED SPECIALIZATIONS =====
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

/// OPTIMIZED: Lean version of QLinearConv with pre-computation and cache optimizations (used for floating point)
pub fn qlinearconv_lean_fp(
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
