const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const tensor_module = zant.core.tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const Allocator = std.mem.Allocator;
const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");

inline fn typeIsInt(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

inline fn valueAsF32(comptime T: type, v: T) f32 {
    return switch (@typeInfo(T)) {
        .float => @as(f32, @floatCast(v)),
        .int, .comptime_int => @as(f32, @floatFromInt(v)),
        else => @compileError("Unsupported type for float cast"),
    };
}
const TensorMathError = zant.utils.error_handler.TensorMathError;
const tensMath = zant.core.tensor.math_standard;

pub var log_functionC: ?*const fn ([*c]u8) callconv(.C) void = null;

pub export fn setLogFunctionC(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_functionC = func;
}

inline fn logDebug(comptime fmt: []const u8, args: anytype) void {
    var log_ptr: ?*const fn ([*c]u8) callconv(.C) void = log_functionC;
    if (log_ptr == null) {
        log_ptr = tensor_module.log_function;
    }
    if (log_ptr) |log| {
        var buffer: [512:0]u8 = undefined;
        const msg = std.fmt.bufPrintZ(&buffer, fmt, args) catch return;
        log(@constCast(msg.ptr));
    }
}

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("op_convolution.zig");

const WeightLayout = enum {
    standard,
    ohwi,
    depthwise_khwc,
};

fn detectWeightLayout(weight_shape: []const usize, in_channels: usize, group: usize) !WeightLayout {
    if (weight_shape.len != 4) {
        logDebug("[ZANT][qlinearconv] invalid weight rank={d}\n", .{weight_shape.len});
        return TensorMathError.InvalidDimensions;
    }

    if (group == 0 or in_channels == 0 or in_channels % group != 0) {
        logDebug(
            "[ZANT][qlinearconv] invalid group/in_channels configuration weight_shape={any} in_channels={d} group={d}\n",
            .{ weight_shape, in_channels, group },
        );
        return TensorMathError.InvalidDimensions;
    }

    if (weight_shape[1] * group == in_channels) {
        logDebug(
            "[ZANT][qlinearconv] detected standard weight layout shape={any} group={d}\n",
            .{ weight_shape, group },
        );
        return .standard;
    }

    if (weight_shape[3] * group == in_channels) {
        logDebug(
            "[ZANT][qlinearconv] detected OHWI weight layout shape={any} group={d}\n",
            .{ weight_shape, group },
        );
        return .ohwi;
    }

    if (group == in_channels and weight_shape[3] == in_channels) {
        logDebug(
            "[ZANT][qlinearconv] detected depthwise KHWC weight layout shape={any} group={d}\n",
            .{ weight_shape, group },
        );
        return .depthwise_khwc;
    }

    logDebug(
        "[ZANT][qlinearconv] unsupported weight layout shape={any} in_channels={d} group={d}\n",
        .{ weight_shape, in_channels, group },
    );
    return TensorMathError.InvalidDimensions;
}

fn normalizedWeightShape(
    layout: WeightLayout,
    weight_shape: []const usize,
) [4]usize {
    return switch (layout) {
        .standard => .{ weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3] },
        .ohwi => .{ weight_shape[0], weight_shape[3], weight_shape[1], weight_shape[2] },
        .depthwise_khwc => .{ weight_shape[0] * weight_shape[3], 1, weight_shape[1], weight_shape[2] },
    };
}

fn inferWeightLayout(
    weight_shape: []const usize,
    in_channels: usize,
    group_hint: ?usize,
) !WeightLayout {
    if (weight_shape.len != 4 or in_channels == 0) {
        return TensorMathError.InvalidDimensions;
    }

    if (group_hint) |g| blk: {
        if (g == 0 or in_channels % g != 0) break :blk;
        if (detectWeightLayout(weight_shape, in_channels, g)) |layout| {
            return layout;
        } else |err| switch (err) {
            TensorMathError.InvalidDimensions => {},
            else => return err,
        }
    }

    var candidate: usize = 1;
    while (candidate <= in_channels) : (candidate += 1) {
        if (in_channels % candidate != 0) continue;
        if (detectWeightLayout(weight_shape, in_channels, candidate)) |layout| {
            return layout;
        } else |err| switch (err) {
            TensorMathError.InvalidDimensions => continue,
            else => return err,
        }
    }

    return TensorMathError.InvalidDimensions;
}

fn ensureWeightTensorLayout(
    comptime WeightType: type,
    weight: *const Tensor(WeightType),
    in_channels: usize,
    group: usize,
) !struct {
    layout: WeightLayout,
    tensor: *const Tensor(WeightType),
    owned: ?Tensor(WeightType),
} {
    const layout = detectWeightLayout(weight.shape, in_channels, group) catch |err| switch (err) {
        TensorMathError.InvalidDimensions => blk: {
            if (group == in_channels and weight.shape.len == 4) {
                const fallback: ?WeightLayout = inferWeightLayout(weight.shape, in_channels, null) catch |infer_err| switch (infer_err) {
                    TensorMathError.InvalidDimensions => null,
                    else => return infer_err,
                };

                if (fallback) |layout_guess| switch (layout_guess) {
                    .depthwise_khwc => {
                        logDebug(
                            "[ZANT][qlinearconv] inferring depthwise layout from fallback detection shape={any}\n",
                            .{weight.shape},
                        );
                        break :blk WeightLayout.depthwise_khwc;
                    },
                    .ohwi => if (weight.shape[0] == 1 and weight.shape[3] == in_channels) {
                        logDebug(
                            "[ZANT][qlinearconv] interpreting OHWI fallback as depthwise layout shape={any}\n",
                            .{weight.shape},
                        );
                        break :blk WeightLayout.depthwise_khwc;
                    },
                    else => {},
                };
            }

            return err;
        },
        else => return err,
    };

    if (layout == .standard) {
        logDebug(
            "[ZANT][qlinearconv] using standard weight tensor shape={any}\n",
            .{weight.shape},
        );
        return .{
            .layout = layout,
            .tensor = weight,
            .owned = null,
        };
    }

    switch (layout) {
        .ohwi => {
            logDebug(
                "[ZANT][qlinearconv] transposing OHWI weights original_shape={any}\n",
                .{weight.shape},
            );
            var transposed_shape = [_]usize{ weight.shape[0], weight.shape[3], weight.shape[1], weight.shape[2] };
            var transposed = try Tensor(WeightType).fromShape(&pkg_allocator, &transposed_shape);
            errdefer transposed.deinit();

            const perm = [_]usize{ 0, 3, 1, 2 };
            try tensMath.transpose_onnx_lean(WeightType, @constCast(weight), perm[0..], &transposed, pkg_allocator);

            logDebug(
                "[ZANT][qlinearconv] transposed weights normalized_shape={any}\n",
                .{transposed_shape},
            );

            return .{
                .layout = layout,
                .tensor = &transposed,
                .owned = transposed,
            };
        },
        .depthwise_khwc => {
            logDebug(
                "[ZANT][qlinearconv] reordering depthwise KHWC weights original_shape={any}\n",
                .{weight.shape},
            );

            const channel_multiplier = weight.shape[0];
            const kernel_h = weight.shape[1];
            const kernel_w = weight.shape[2];
            const input_channels = weight.shape[3];

            var normalized_shape = [_]usize{ channel_multiplier * input_channels, 1, kernel_h, kernel_w };
            var reordered = try Tensor(WeightType).fromShape(&pkg_allocator, &normalized_shape);
            errdefer reordered.deinit();

            var in_channel: usize = 0;
            while (in_channel < input_channels) : (in_channel += 1) {
                var multiplier: usize = 0;
                while (multiplier < channel_multiplier) : (multiplier += 1) {
                    const out_channel = in_channel * channel_multiplier + multiplier;
                    var h: usize = 0;
                    while (h < kernel_h) : (h += 1) {
                        var w: usize = 0;
                        while (w < kernel_w) : (w += 1) {
                            const src_index = ((((multiplier * kernel_h) + h) * kernel_w) + w) * input_channels + in_channel;
                            const dst_index = (((out_channel * kernel_h) + h) * kernel_w) + w;
                            reordered.data[dst_index] = weight.data[src_index];
                        }
                    }
                }
            }

            logDebug(
                "[ZANT][qlinearconv] reordered depthwise weights normalized_shape={any}\n",
                .{normalized_shape},
            );

            return .{
                .layout = layout,
                .tensor = &reordered,
                .owned = reordered,
            };
        },
        else => unreachable,
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
    return applyAsymmetricPaddingWithAllocator(InputType, original, pad_top, pad_left, pad_bottom, pad_right, fill_zero_point, &pkg_allocator);
}

fn applyAsymmetricPaddingWithAllocator(
    comptime InputType: type,
    original: *const Tensor(InputType),
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    fill_zero_point: i32,
    allocator: *const Allocator,
) !Tensor(InputType) {
    const batch = original.shape[0];
    const channels = original.shape[1];
    const in_height = original.shape[2];
    const in_width = original.shape[3];

    const padded_height = in_height + pad_top + pad_bottom;
    const padded_width = in_width + pad_left + pad_right;
    var padded_shape = [_]usize{ batch, channels, padded_height, padded_width };

    var padded = try Tensor(InputType).fromShape(allocator, &padded_shape);
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

    const actual_group = group orelse 1;
    const layout = try detectWeightLayout(w.shape, input_shape[1], actual_group);
    const normalized_weight_shape = normalizedWeightShape(layout, w.shape);
    const weight_shape_slice: []const usize = switch (layout) {
        .standard => w.shape,
        .ohwi, .depthwise_khwc => normalized_weight_shape[0..],
    };

    // Calculate output shape using existing conv calculation
    const output_shape = try conv.calculateOutputShape(InputType, &input_shape, weight_shape_slice, stride, pads, dilations, auto_pad);

    const auto_pad_slice = auto_pad orelse "<null>";
    logDebug(
        "[ZANT][qlinearconv] qlinearconv entry x_shape={any} w_shape={any} y_shape={any} group={d} auto_pad={s}\n",
        .{ input_ptr.shape, w.shape, output_shape, actual_group, auto_pad_slice },
    );

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, &output_shape);
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

    const actual_group = group orelse 1;
    const in_channels = x.shape[1];
    var normalized_weight = try ensureWeightTensorLayout(WeightType, w, in_channels, actual_group);
    defer if (normalized_weight.owned) |*tmp| tmp.deinit();
    const weight_tensor = normalized_weight.tensor;
    const weight_shape = weight_tensor.shape;
    const weight_data = weight_tensor.data;

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
    const in_height = x.shape[2]; // H
    const in_width = x.shape[3]; // W

    const out_channels = weight_shape[0]; // M
    const weight_in_channels = weight_shape[1]; // C/group
    const kernel_height = weight_shape[2]; // kH
    const kernel_width = weight_shape[3]; // kW

    const out_height = output.shape[2]; // oH
    const out_width = output.shape[3]; // oW

    // Parametri
    // actual_group already computed above
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
        try conv3x3Optimized(x, weight_tensor, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else if (kernel_height == 1 and kernel_width == 1) {
        // Ottimizzato per 1x1 (pointwise)
        try conv1x1Optimized(x, weight_tensor, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
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
                                                const qw = asF32(WeightType, weight_data[weight_idx]);
                                                break :blk w_scale_val * (qw - w_zp_f);
                                            } else asF32(WeightType, weight_data[weight_idx]);

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

fn qlinearconv_minimal(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
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
    if (auto_pad.len != 0 and !std.mem.eql(u8, auto_pad, "NOTSET")) {
        return TensorMathError.InvalidPadding;
    }

    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const actual_group = group orelse 1;
    const in_channels = x.shape[1];
    var normalized_weight = try ensureWeightTensorLayout(WeightType, w, in_channels, actual_group);
    defer if (normalized_weight.owned) |*tmp| tmp.deinit();
    const weight_tensor = normalized_weight.tensor;
    const weight_shape = weight_tensor.shape;
    const weight_data = weight_tensor.data;

    const batch_size = x.shape[0];
    const in_height = x.shape[2];
    const in_width = x.shape[3];

    const out_channels = weight_shape[0];
    const weight_in_channels = weight_shape[1];
    const kernel_height = weight_shape[2];
    const kernel_width = weight_shape[3];

    const out_height = output.shape[2];
    const out_width = output.shape[3];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h_begin = pads_arr[0];
    const pad_w_begin = pads_arr[1];
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    if (in_channels % actual_group != 0 or out_channels % actual_group != 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * actual_group != in_channels) {
        return TensorMathError.InvalidDimensions;
    }

    const x_scale_val = valueAsF32(ScaleType, x_scale.data[0]);
    const y_scale_val = valueAsF32(ScaleType, y_scale.data[0]);
    const input_zero_point = readScalarZP(InputType, x_zero_point);
    const output_zero_point = readScalarZP(InputType, y_zero_point);
    const x_zp_f = @as(f32, @floatFromInt(input_zero_point));
    const y_zp_f = @as(f32, @floatFromInt(output_zero_point));

    const input_is_int = typeIsInt(InputType);
    const weight_is_int = typeIsInt(WeightType);
    const bias_tensor = if (bias) |b| b else null;
    const bias_is_int = typeIsInt(BiasType);

    const q_min: f32 = if (input_is_int)
        valueAsF32(InputType, std.math.minInt(InputType))
    else
        -std.math.inf(f32);
    const q_max: f32 = if (input_is_int)
        valueAsF32(InputType, std.math.maxInt(InputType))
    else
        std.math.inf(f32);

    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                const w_scale_val = readPerChannelScale(ScaleType, w_scale, m, out_channels);
                const w_zero = readPerChannelZP(w_zero_point, m, out_channels);
                const w_zp_f = @as(f32, @floatFromInt(w_zero));
                const bias_f: f32 = if (bias_tensor) |b_tensor| blk: {
                    if (b_tensor.data.len == 0) break :blk 0.0;
                    const idx = if (b_tensor.data.len == 1)
                        0
                    else
                        selectChannelIndex(b_tensor.data.len, m);
                    const raw = b_tensor.data[idx];
                    if (bias_is_int) {
                        break :blk valueAsF32(BiasType, raw) * x_scale_val * w_scale_val;
                    }
                    break :blk valueAsF32(BiasType, raw);
                } else 0.0;

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

                                    const x_val = valueAsF32(InputType, x.data[input_idx]);
                                    const w_val = valueAsF32(WeightType, weight_data[weight_idx]);

                                    const x_real: f32 = if (input_is_int)
                                        x_scale_val * (x_val - x_zp_f)
                                    else
                                        x_val;
                                    const w_real: f32 = if (weight_is_int)
                                        w_scale_val * (w_val - w_zp_f)
                                    else
                                        w_val;

                                    acc += x_real * w_real;
                                }
                            }
                        }

                        const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                        if (input_is_int) {
                            const q_unrounded = acc / y_scale_val + y_zp_f;
                            const q_rounded = if (q_unrounded >= 0)
                                @floor(q_unrounded + 0.5)
                            else
                                @ceil(q_unrounded - 0.5);
                            const q_clamped = std.math.clamp(q_rounded, q_min, q_max);
                            output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                        } else {
                            output.data[output_idx] = @as(InputType, @floatCast(acc));
                        }
                    }
                }
            }
        }
    }
}

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
    const Impl = struct {
        fn run(
            comptime InputTypeInner: anytype,
            comptime WeightTypeInner: anytype,
            comptime ScaleTypeInner: anytype,
            comptime BiasTypeInner: anytype,
            x_inner: *const Tensor(InputTypeInner),
            x_scale_inner: *const Tensor(ScaleTypeInner),
            x_zero_point_inner: anytype,
            w_inner: *const Tensor(WeightTypeInner),
            w_scale_inner: *const Tensor(ScaleTypeInner),
            w_zero_point_inner: anytype,
            output_inner: *Tensor(InputTypeInner),
            y_scale_inner: *const Tensor(ScaleTypeInner),
            y_zero_point_inner: anytype,
            bias_inner: ?*const Tensor(BiasTypeInner),
            stride_inner: ?[]const usize,
            pads_inner: ?[]const usize,
            dilations_inner: ?[]const usize,
            group_inner: ?usize,
            auto_pad_inner: []const u8,
        ) !void {
            if (auto_pad_inner.len != 0 and !std.mem.eql(u8, auto_pad_inner, "NOTSET")) {
                return TensorMathError.InvalidPadding;
            }

            if (!typeIsInt(InputTypeInner) or !typeIsInt(WeightTypeInner)) {
                return qlinearconv_lean(
                    InputTypeInner,
                    WeightTypeInner,
                    ScaleTypeInner,
                    void,
                    BiasTypeInner,
                    x_inner,
                    x_scale_inner,
                    x_zero_point_inner,
                    w_inner,
                    w_scale_inner,
                    w_zero_point_inner,
                    output_inner,
                    y_scale_inner,
                    y_zero_point_inner,
                    bias_inner,
                    stride_inner,
                    pads_inner,
                    dilations_inner,
                    group_inner,
                    auto_pad_inner,
                );
            }

            if (x_inner.shape.len != 4 or w_inner.shape.len != 4 or output_inner.shape.len != 4) {
                return TensorMathError.InvalidDimensions;
            }

            const actual_group_inner = group_inner orelse 1;
            const in_channels_inner = x_inner.shape[1];
            var normalized_weight_inner = try ensureWeightTensorLayout(WeightTypeInner, w_inner, in_channels_inner, actual_group_inner);
            defer if (normalized_weight_inner.owned) |*tmp| tmp.deinit();
            const weight_shape_inner = normalized_weight_inner.tensor.shape;

            const batch_size_inner = x_inner.shape[0];
            const in_height_inner = x_inner.shape[2];
            const in_width_inner = x_inner.shape[3];

            const out_channels_inner = weight_shape_inner[0];
            const weight_in_channels_inner = weight_shape_inner[1];
            const kernel_height_inner = weight_shape_inner[2];
            const kernel_width_inner = weight_shape_inner[3];

            const out_height_inner = output_inner.shape[2];
            const out_width_inner = output_inner.shape[3];

            const stride_h_inner = if (stride_inner) |s| s[0] else 1;
            const stride_w_inner = if (stride_inner) |s| (if (s.len > 1) s[1] else s[0]) else 1;
            const pads_arr_inner = pads_inner orelse &[_]usize{ 0, 0, 0, 0 };
            const pad_h_begin_inner = pads_arr_inner[0];
            const pad_w_begin_inner = pads_arr_inner[1];
            const dilation_h_inner = if (dilations_inner) |d| d[0] else 1;
            const dilation_w_inner = if (dilations_inner) |d| (if (d.len > 1) d[1] else d[0]) else 1;
            if (in_channels_inner % actual_group_inner != 0 or out_channels_inner % actual_group_inner != 0) {
                return TensorMathError.InvalidDimensions;
            }
            if (weight_in_channels_inner * actual_group_inner != in_channels_inner) {
                return TensorMathError.InvalidDimensions;
            }

            const SCALE_SHIFT: u5 = 16;
            const scale_factor_inner = @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT));
            const x_scale_val_inner = valueAsF32(ScaleTypeInner, x_scale_inner.data[0]);
            const x_scale_q16_inner = @as(i64, q16(x_scale_val_inner));
            const y_scale_val_inner = valueAsF32(ScaleTypeInner, y_scale_inner.data[0]);
            const input_zero_point_inner = if (@typeInfo(@TypeOf(x_zero_point_inner)) == .pointer and x_zero_point_inner.data.len == 0)
                0
            else
                readScalarZP(InputTypeInner, x_zero_point_inner);
            const output_zero_point_inner = if (@typeInfo(@TypeOf(y_zero_point_inner)) == .pointer and y_zero_point_inner.data.len == 0)
                0
            else
                readScalarZP(InputTypeInner, y_zero_point_inner);

            const quant_inner = QuantParams{
                .scale_shift = SCALE_SHIFT,
                .input_zero_point = input_zero_point_inner,
                .output_inv_scale_i64 = @as(i64, q16(1.0 / y_scale_val_inner)),
                .output_zero_point_q16 = @as(i64, output_zero_point_inner) << SCALE_SHIFT,
                .q_min = @as(i32, @intCast(std.math.minInt(InputTypeInner))),
                .q_max = @as(i32, @intCast(std.math.maxInt(InputTypeInner))),
            };

            const dims_inner = ConvDims{
                .batch = batch_size_inner,
                .in_channels = in_channels_inner,
                .in_height = in_height_inner,
                .in_width = in_width_inner,
                .out_channels = out_channels_inner,
                .out_height = out_height_inner,
                .out_width = out_width_inner,
                .kernel_height = kernel_height_inner,
                .kernel_width = kernel_width_inner,
                .stride_h = stride_h_inner,
                .stride_w = stride_w_inner,
                .pad_h = pad_h_begin_inner,
                .pad_w = pad_w_begin_inner,
                .dilation_h = dilation_h_inner,
                .dilation_w = dilation_w_inner,
                .groups = actual_group_inner,
                .group_in_channels = in_channels_inner / actual_group_inner,
                .group_out_channels = out_channels_inner / actual_group_inner,
            };

            const layout_inner = ConvLayout{
                .input_batch_stride = in_channels_inner * in_height_inner * in_width_inner,
                .input_channel_stride = in_height_inner * in_width_inner,
                .input_row_stride = in_width_inner,
                .weight_out_stride = weight_in_channels_inner * kernel_height_inner * kernel_width_inner,
                .weight_channel_stride = kernel_height_inner * kernel_width_inner,
                .weight_row_stride = kernel_width_inner,
                .output_batch_stride = out_channels_inner * out_height_inner * out_width_inner,
                .output_channel_stride = out_height_inner * out_width_inner,
                .output_row_stride = out_width_inner,
            };

            var channel_params_inner = try pkg_allocator.alloc(ChannelParams, out_channels_inner);
            defer pkg_allocator.free(channel_params_inner);

            const bias_tensor_inner = if (bias_inner) |b| b else null;
            const bias_is_int_inner = typeIsInt(BiasTypeInner);

            for (0..out_channels_inner) |m| {
                const w_scale_val_inner: f32 = if (w_scale_inner.data.len == out_channels_inner)
                    valueAsF32(ScaleTypeInner, w_scale_inner.data[m])
                else
                    valueAsF32(ScaleTypeInner, w_scale_inner.data[0]);

                const weight_scale_q16_inner = @as(i64, q16(w_scale_val_inner));
                const combined_scale_q16_inner = saturateI128ToI64((@as(i128, x_scale_q16_inner) * @as(i128, weight_scale_q16_inner)) >> SCALE_SHIFT);
                const weight_zero_point_inner = if (@typeInfo(@TypeOf(w_zero_point_inner)) == .pointer and w_zero_point_inner.data.len == 0)
                    0
                else
                    readPerChannelZP(w_zero_point_inner, m, out_channels_inner);

                const bias_q16_inner = if (bias_tensor_inner) |b_tensor| blk: {
                    if (b_tensor.data.len == 0) break :blk 0;
                    const raw = if (b_tensor.data.len == 1) b_tensor.data[0] else b_tensor.data[m];
                    var bias_real = valueAsF32(BiasTypeInner, raw);
                    if (bias_is_int_inner) {
                        bias_real *= x_scale_val_inner * w_scale_val_inner;
                    }
                    break :blk @as(i64, @intFromFloat(@round(bias_real * scale_factor_inner)));
                } else 0;

                channel_params_inner[m] = ChannelParams{
                    .combined_scale_q16 = combined_scale_q16_inner,
                    .weight_zero_point = weight_zero_point_inner,
                    .bias_q16 = bias_q16_inner,
                };
            }

            if (kernel_height_inner == 3 and kernel_width_inner == 3 and dilation_h_inner == 1 and dilation_w_inner == 1) {
                conv3x3EmbeddedOptimized(InputTypeInner, WeightTypeInner, x_inner.data, normalized_weight_inner.tensor.data, output_inner.data, dims_inner, layout_inner, quant_inner, channel_params_inner);
            } else if (kernel_height_inner == 1 and kernel_width_inner == 1) {
                conv1x1EmbeddedOptimized(InputTypeInner, WeightTypeInner, x_inner.data, normalized_weight_inner.tensor.data, output_inner.data, dims_inner, layout_inner, quant_inner, channel_params_inner);
            } else {
                convGenericEmbeddedOptimized(InputTypeInner, WeightTypeInner, x_inner.data, normalized_weight_inner.tensor.data, output_inner.data, dims_inner, layout_inner, quant_inner, channel_params_inner);
            }
        }
    };

    return Impl.run(
        InputType,
        WeightType,
        ScaleType,
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
    ) catch |err| switch (err) {
        error.OutOfMemory => {
            logDebug("[ZANT][qlinearconv] embedded fallback to minimal due to OutOfMemory\n", .{});
            return qlinearconv_minimal(
                InputType,
                WeightType,
                ScaleType,
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
        },
        else => return err,
    };
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

inline fn fallbackToEmbedded(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
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
        w_zero_point_any,
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

const ChunkError = error{
    CmsisKernelFailed,
};

fn runCmsisStandardConvChunked(
    comptime InputType: anytype,
    alloc: Allocator,
    output: *Tensor(InputType),
    input_ptr_s8: [*]const i8,
    conv_params: *const cmsis_nn.ConvParams,
    input_dims: *const cmsis_nn.Dims,
    filter_dims_full: cmsis_nn.Dims,
    bias_dims_full: cmsis_nn.Dims,
    output_dims_full: cmsis_nn.Dims,
    batch_size: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,
    weight_ptr_s8: [*]const i8,
    weight_in_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    multipliers_buf: []i32,
    shifts_buf: []i32,
    bias_ptr: ?[*]const i32,
    ctx: *const cmsis_nn.Context,
    ctx_size_i32: i32,
    is_u8_input: bool,
    zero_restore: i32,
) !void {
    const pixels_per_channel = batch_size * out_height * out_width;
    if (pixels_per_channel == 0 or out_channels == 0) {
        logDebug("[ZANT][qlinearconv][CMSIS][chunk] trivial output (no pixels or channels)\n", .{});
        return;
    }

    var chunk_channels_try: usize = out_channels;
    var chunk_buf_opt: ?[]i8 = null;
    defer if (chunk_buf_opt) |buf| alloc.free(buf);
    var chunk_capacity_channels: usize = 0;
    while (chunk_channels_try > 0) {
        const chunk_elems = pixels_per_channel * chunk_channels_try;
        logDebug(
            "[ZANT][qlinearconv][CMSIS][chunk] ensure output scratch for {d} channels ({d} bytes)\n",
            .{ chunk_channels_try, chunk_elems },
        );
        const attempt = alloc.alloc(i8, chunk_elems) catch {
            chunk_channels_try = chunk_channels_try / 2;
            continue;
        };
        chunk_buf_opt = attempt;
        chunk_capacity_channels = chunk_channels_try;
        break;
    }

    if (chunk_capacity_channels == 0) {
        logDebug("[ZANT][qlinearconv][CMSIS][chunk] unable to allocate any output scratch\n", .{});
        return error.OutOfMemory;
    }

    const chunk_buf = chunk_buf_opt.?;

    if (chunk_capacity_channels < out_channels) {
        logDebug(
            "[ZANT][qlinearconv][CMSIS][chunk] falling back to channel chunks capacity={d} total={d}\n",
            .{ chunk_capacity_channels, out_channels },
        );
    } else {
        logDebug("[ZANT][qlinearconv][CMSIS][chunk] chunk capacity covers full output\n", .{});
    }

    const out_spatial = out_height * out_width;
    const out_batch_stride = out_spatial * out_channels;

    var channel_base: usize = 0;
    while (channel_base < out_channels) {
        const remaining = out_channels - channel_base;
        const channels_this = if (remaining < chunk_capacity_channels) remaining else chunk_capacity_channels;
        const chunk_elems = pixels_per_channel * channels_this;
        var chunk_slice = chunk_buf[0..chunk_elems];

        var chunk_filter_dims = filter_dims_full;
        chunk_filter_dims.n = @intCast(channels_this);

        var chunk_bias_dims = bias_dims_full;
        chunk_bias_dims.c = @intCast(channels_this);

        var chunk_output_dims = output_dims_full;
        chunk_output_dims.c = @intCast(channels_this);

        var chunk_quant_params = cmsis_nn.PerChannelQuantParams{
            .multiplier = multipliers_buf[channel_base .. channel_base + channels_this].ptr,
            .shift = shifts_buf[channel_base .. channel_base + channels_this].ptr,
        };

        const weights_offset = channel_base * kernel_height * kernel_width * weight_in_channels;
        const chunk_weight_ptr = weight_ptr_s8 + weights_offset;
        const chunk_bias_ptr = if (bias_ptr) |ptr| ptr + channel_base else null;

        var wrap_buf_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(
            conv_params,
            input_dims,
            &chunk_filter_dims,
            &chunk_output_dims,
        );
        if (wrap_buf_size < 0) wrap_buf_size = 0;

        var wrap_ctx = ctx.*;
        var wrap_dyn: ?[]u8 = null;
        defer if (wrap_dyn) |b| alloc.free(b);
        if (wrap_buf_size > ctx_size_i32) {
            const newb = try alloc.alignedAlloc(u8, @alignOf(i32), @intCast(wrap_buf_size));
            wrap_dyn = newb;
            wrap_ctx = .{ .buf = newb.ptr, .size = @intCast(newb.len) };
        }

        const status = cmsis_nn.conv.arm_convolve_wrapper_s8(
            &wrap_ctx,
            conv_params,
            &chunk_quant_params,
            input_dims,
            input_ptr_s8,
            &chunk_filter_dims,
            chunk_weight_ptr,
            &chunk_bias_dims,
            if (chunk_bias_ptr) |ptr| @ptrCast(ptr) else null,
            &chunk_output_dims,
            chunk_slice.ptr,
        );
        if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
            logDebug(
                "[ZANT][qlinearconv][CMSIS][chunk] wrapper failed status={d} base={d} count={d}\n",
                .{ status, channel_base, channels_this },
            );
            return ChunkError.CmsisKernelFailed;
        }

        var src_batch_base: usize = 0;
        var n_back: usize = 0;
        while (n_back < batch_size) : (n_back += 1) {
            var pixel: usize = 0;
            while (pixel < out_spatial) : (pixel += 1) {
                var src_idx = src_batch_base + pixel * channels_this;
                const dst_base = n_back * out_batch_stride + pixel;
                var c: usize = 0;
                while (c < channels_this) : (c += 1) {
                    const dst_idx = dst_base + (channel_base + c) * out_spatial;
                    const v_i32 = @as(i32, @intCast(chunk_slice[src_idx]));
                    const adjusted = v_i32 + zero_restore;
                    if (is_u8_input) {
                        output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
                    } else {
                        output.data[dst_idx] = @as(InputType, @intCast(adjusted));
                    }
                    src_idx += 1;
                }
            }
            src_batch_base += out_spatial * channels_this;
        }

        channel_base += channels_this;
    }
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
    const actual_group = group orelse 1;
    if (x.shape.len < 2) {
        logDebug("[ZANT][qlinearconv] dispatch invalid input rank={d}\n", .{x.shape.len});
        return TensorMathError.InvalidDimensions;
    }

    const accelerators = @import("../Accelerators/mod.zig");
    if (!accelerators.canUseCmsisHelium()) {
        logDebug("[ZANT][qlinearconv] dispatch selecting embedded implementation\n", .{});
        // Reference build: force embedded fixed-point implementation
        const in_channels = x.shape[1];
        var normalized_weight = try ensureWeightTensorLayout(WeightType, w, in_channels, actual_group);
        defer if (normalized_weight.owned) |*tmp| tmp.deinit();
        const weight_ptr = normalized_weight.tensor;
        logDebug(
            "[ZANT][qlinearconv] dispatch config (embedded) x_shape={any} w_shape={any} y_shape={any} group={d} auto_pad={s}\n",
            .{ x.shape, weight_ptr.shape, output.shape, actual_group, auto_pad },
        );
        return qlinearconv_embedded_lean(
            InputType,
            WeightType,
            ScaleType,
            void,
            BiasType,
            x,
            x_scale,
            x_zero_point,
            weight_ptr,
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
    logDebug("[ZANT][qlinearconv] dispatch selecting CMSIS implementation\n", .{});
    logDebug(
        "[ZANT][qlinearconv] dispatch config (cmsis) x_shape={any} w_shape={any} y_shape={any} group={d} auto_pad={s}\n",
        .{ x.shape, w.shape, output.shape, actual_group, auto_pad },
    );
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
    ) catch |err| switch (err) {
        error.OutOfMemory => {
            logDebug("[ZANT][qlinearconv] CMSIS error OutOfMemory (no fallback)\n", .{});
            return err;
        },
        else => return err,
    };
}

/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
fn qlinearconv_cmsis_accelerated_impl(
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
    allocator: *const Allocator,
) !void {
    // Mark CMSIS usage for testing
    const accelerators = @import("../Accelerators/mod.zig");
    accelerators.markCmsisUsed();

    // DEBUG: Print function entry
    // std.debug.print("CMSIS DEBUG: qlinearconv_cmsis_accelerated called\n", .{});

    // Use allocator for transient buffers
    var alloc = allocator.*;

    // Basic validation
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    // allocator currently unused after workspace refactor

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    // Guard: CMSIS-NN conv s8 path (generic and wrappers) does not support dilation != 1 on many kernels.
    // Fallback to embedded reference when dilation is used to avoid runtime errors (rc=-1).
    if (dilation_h != 1 or dilation_w != 1) {
        logDebug(
            "[ZANT][qlinearconv] CMSIS fallback to embedded due to dilation h={d} w={d}\n",
            .{ dilation_h, dilation_w },
        );
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

    // Extract dimensions (assume NCHW tensors)
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    var in_height = x.shape[2];
    var in_width = x.shape[3];
    const out_channels = output.shape[1];
    const weight_shape = w.shape;
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

    // If padding is asymmetric, integrate it into the NHWC conversion buffer to avoid allocating a padded tensor
    const orig_in_height = in_height;
    const orig_in_width = in_width;
    var conv_pad_h: usize = pad_h;
    var conv_pad_w: usize = pad_w;
    // Force symmetric pad at CMSIS level for stride>1 without growing input buffer
    var extra_top: usize = 0;
    var extra_left: usize = 0;
    if (pad_bottom != pad_h or pad_right != pad_w) {
        // Use max(top,bottom), max(left,right) as CMSIS padding and keep input dims unchanged
        const sym_h = if (pad_h > pad_bottom) pad_h else pad_bottom;
        const sym_w = if (pad_w > pad_right) pad_w else pad_right;
        logDebug(
            "[ZANT][qlinearconv] CMSIS pad fix: original tlbr={d},{d},{d},{d} -> sym={d},{d}\n",
            .{ pad_h, pad_w, pad_bottom, pad_right, sym_h, sym_w },
        );
        conv_pad_h = sym_h;
        conv_pad_w = sym_w;
        // No changes to in_height/in_width; no buffer enlargement
        extra_top = 0;
        extra_left = 0;
    }

    const input_zero_point = readScalarZP(InputType, x_zero_point);
    const output_zero_point = readScalarZP(InputType, y_zero_point);
    logDebug(
        "[ZANT][qlinearconv] CMSIS zero points input={d} output={d}\n",
        .{ input_zero_point, output_zero_point },
    );

    logDebug(
        "[ZANT][qlinearconv] CMSIS config x_shape={any} w_shape={any} y_shape={any} stride={d}x{d} pads={d},{d},{d},{d} group={d} auto_pad={s}\n",
        .{ x.shape, w.shape, output.shape, stride_h, stride_w, pad_h, pad_w, pad_bottom, pad_right, group_val, auto_pad },
    );

    // Integrate asymmetric padding logically into NHWC conversion to avoid allocating a padded tensor
    // Make the effective input spatial match (H + top + bottom, W + left + right), and disable CMSIS padding.
    if (pad_bottom != pad_h or pad_right != pad_w) {
        in_height = in_height + pad_h + pad_bottom;
        in_width = in_width + pad_w + pad_right;
        pad_h = 0;
        pad_w = 0;
        conv_pad_h = 0;
        conv_pad_w = 0;
    }

    const effective_input: *const Tensor(InputType) = x;

    if (group_val == 0) {
        logDebug("[ZANT][qlinearconv] CMSIS invalid group 0\n", .{});
        return TensorMathError.InvalidDimensions;
    }
    // Depthwise after OHWI conversion should have weights shaped [1, kH, kW, M]
    const is_depthwise = (group_val == in_channels) and (weight_shape[0] == 1);
    if (out_channels % group_val != 0) {
        logDebug(
            "[ZANT][qlinearconv] CMSIS mismatch out_channels={d} group={d}\n",
            .{ out_channels, group_val },
        );
        return TensorMathError.InvalidDimensions;
    }

    var kernel_height: usize = undefined;
    var kernel_width: usize = undefined;
    var group_in_channels: usize = undefined;
    var weight_in_channels: usize = undefined;
    const group_out_channels = out_channels / group_val;

    if (is_depthwise) {
        // Expect OHWI layout for depthwise: [1, kH, kW, M]
        if (weight_shape[0] != 1 or weight_shape[3] != out_channels) {
            logDebug(
                "[ZANT][qlinearconv] CMSIS depthwise weight shape invalid w_shape={any} oc={d}\n",
                .{ weight_shape, out_channels },
            );
            return TensorMathError.InvalidDimensions;
        }
        kernel_height = weight_shape[1];
        kernel_width = weight_shape[2];
        group_in_channels = 1;
        weight_in_channels = 1;
    } else {
        // Expect OHWI layout: [M, kH, kW, C/group]
        if (weight_shape[0] != out_channels) {
            logDebug(
                "[ZANT][qlinearconv] CMSIS mismatch weight_dim0={d} out_channels={d}\n",
                .{ weight_shape[0], out_channels },
            );
            return TensorMathError.InvalidDimensions;
        }
        kernel_height = weight_shape[1];
        kernel_width = weight_shape[2];
        group_in_channels = weight_shape[3];
        weight_in_channels = group_in_channels;
        if (group_in_channels * group_val != in_channels) {
            logDebug(
                "[ZANT][qlinearconv] CMSIS mismatch group_in_channels={d} group={d} in_channels={d}\n",
                .{ group_in_channels, group_val, in_channels },
            );
            return TensorMathError.InvalidDimensions;
        }
    }

    const expected_weights: usize = out_channels * group_in_channels * kernel_height * kernel_width;
    if (w.data.len != expected_weights) {
        logDebug(
            "[ZANT][qlinearconv] CMSIS weight size mismatch have={d} expected={d}\n",
            .{ w.data.len, expected_weights },
        );
        return TensorMathError.InvalidDimensions;
    }

    if (@sizeOf(WeightType) != 1) {
        logDebug("[ZANT][qlinearconv] CMSIS unsupported weight type size={d}\n", .{@sizeOf(WeightType)});
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
    logDebug(
        "[ZANT][qlinearconv] CMSIS scales x={d:.6} y={d:.6} weight_count={d}\n",
        .{ x_scale_val, y_scale_val, w_scale_data.len },
    );

    // CMSIS expects shifts within [-31, 31] and multiplier in Q31. We'll compute robustly.
    const multipliers_buf = try alloc.alloc(i32, out_channels);
    defer alloc.free(multipliers_buf);
    const shifts_buf = try alloc.alloc(i32, out_channels);
    defer alloc.free(shifts_buf);
    logDebug("[ZANT][qlinearconv][CMSIS][alloc] per-channel params bytes mult={d} shifts={d}\n", .{ out_channels * @sizeOf(i32), out_channels * @sizeOf(i32) });

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
    const zero_restore: i32 = if (is_u8_input) 128 else 0;

    // CMSIS s8 uses: input_offset = -x_zp_s8, output_offset = y_zp_s8
    // Clamp to i32 to avoid UB with large zero-points
    const cmsis_input_offset: i32 = -@as(i32, @intCast(input_zero_point_s8));
    const cmsis_output_offset: i32 = @as(i32, @intCast(output_zero_point_s8));

    var conv_params = cmsis_nn.ConvParams{
        .input_offset = cmsis_input_offset,
        .output_offset = cmsis_output_offset,
        .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
        .padding = .{ .h = @intCast(conv_pad_h), .w = @intCast(conv_pad_w) },
        .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
        // CMSIS s8 kernels clamp in s8 domain
        .activation = .{ .min = -128, .max = 127 },
    };
    // Debug: expected out dims with current effective input/padding
    const eff_kh = kernel_height * dilation_h - (dilation_h - 1);
    const eff_kw = kernel_width * dilation_w - (dilation_w - 1);
    const exp_oh_num = @as(isize, @intCast(in_height)) + 2 * @as(isize, @intCast(conv_pad_h)) - @as(isize, @intCast(eff_kh));
    const exp_ow_num = @as(isize, @intCast(in_width)) + 2 * @as(isize, @intCast(conv_pad_w)) - @as(isize, @intCast(eff_kw));
    const exp_oh = @as(isize, @intCast(@divTrunc(exp_oh_num, @as(isize, @intCast(stride_h))))) + 1;
    const exp_ow = @as(isize, @intCast(@divTrunc(exp_ow_num, @as(isize, @intCast(stride_w))))) + 1;
    logDebug("[ZANT][qlinearconv][CMSIS] eff_in={d}x{d} pad={d},{d} stride={d}x{d} exp_out={d}x{d} (tensor_out={d}x{d})\n", .{ in_height, in_width, conv_pad_h, conv_pad_w, stride_h, stride_w, exp_oh, exp_ow, out_height, out_width });

    var quant_params = cmsis_nn.PerChannelQuantParams{
        .multiplier = multipliers_buf.ptr,
        .shift = shifts_buf.ptr,
    };

    // Convert bias to i32 format as expected by CMSIS-NN
    var bias_converted: ?[]i32 = null;
    var bias_ptr: ?[*]const i32 = null;

    if (bias) |b| {
        const bias_buf = try alloc.alloc(i32, out_channels);
        defer alloc.free(bias_buf);
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

    const weight_ptr_s8: [*]const i8 = @ptrCast(w.data.ptr);

    // Allocate buffer required by CMSIS-NN wrapper (regular or depthwise)
    // CMSIS expects OHWI; in their Dims the mapping is filter(n=out, h, w, c=in)
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
    logDebug("[ZANT][qlinearconv][CMSIS] wrapper scratch need={d} bytes (pre-alloc)\n", .{buffer_size});
    var dyn_buffer: ?[]u8 = null;
    defer if (dyn_buffer) |buf| alloc.free(buf);
    var buffer_ptr: ?*anyopaque = null;
    var ctx_size_i32: i32 = 0;
    if (buffer_size > 0) {
        // CMSIS-NN scratch typically needs 4-byte alignment
        const buf = try alloc.alignedAlloc(u8, @alignOf(i32), @intCast(buffer_size));
        dyn_buffer = buf;
        buffer_ptr = buf.ptr;
        ctx_size_i32 = @intCast(buf.len);
        logDebug("[ZANT][qlinearconv][CMSIS][alloc] ctx scratch={d} bytes\n", .{ctx_size_i32});
    }

    var ctx = cmsis_nn.Context{ .buf = buffer_ptr, .size = ctx_size_i32 };
    // Wrapper API does not use upscale_dims

    // Prepare input/output pointers in s8 NHWC domain
    // We assume the Python pre-pass ensured OHWI weights; we still need NHWC input for CMSIS.
    // Avoid extra alloc if we can convert in-place to a scratch, but keep one-time alloc here.
    var input_converted: ?[]i8 = null;
    defer if (input_converted) |buf| alloc.free(buf);
    var output_converted: ?[]i8 = null;
    defer if (output_converted) |buf| alloc.free(buf);
    var grouped_input: ?[]i8 = null;
    defer if (grouped_input) |buf| alloc.free(buf);
    var grouped_output: ?[]i8 = null;
    defer if (grouped_output) |buf| alloc.free(buf);

    // Special low-memory streaming path for 1x1 pointwise conv (group=1, stride=1, pad=0, dilation=1)
    if (group_val == 1 and kernel_height == 1 and kernel_width == 1 and stride_h == 1 and stride_w == 1 and conv_pad_h == 0 and conv_pad_w == 0 and dilation_h == 1 and dilation_w == 1) streaming: {
        logDebug("[ZANT][qlinearconv][CMSIS][1x1-stream] row-wise streaming\n", .{});
        var status_stream = cmsis_nn.ARM_CMSIS_NN_SUCCESS;

        var input_dims_row = cmsis_nn.Dims{ .n = 1, .h = 1, .w = @intCast(in_width), .c = @intCast(in_channels) };
        var output_dims_row = cmsis_nn.Dims{ .n = 1, .h = 1, .w = @intCast(out_width), .c = @intCast(out_channels) };
        var conv_params_row = cmsis_nn.ConvParams{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .stride = .{ .h = 1, .w = 1 },
            .padding = .{ .h = 0, .w = 0 },
            .dilation = .{ .h = 1, .w = 1 },
            .activation = .{ .min = -128, .max = 127 },
        };
        var ctx_row = cmsis_nn.Context{ .buf = null, .size = 0 };

        const row_in_len = in_width * in_channels;
        var row_in_opt: ?[]i8 = null;
        defer if (row_in_opt) |buf| alloc.free(buf);
        row_in_opt = alloc.alloc(i8, row_in_len) catch |err| switch (err) {
            error.OutOfMemory => {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS][1x1-stream] row_in scratch {d} bytes failed -> disable streaming\n",
                    .{ row_in_len * @sizeOf(i8) },
                );
                break :streaming;
            },
            else => return err,
        };
        const row_in = row_in_opt.?;
        const row_out_len = out_width * out_channels;
        var row_out_opt: ?[]i8 = null;
        defer if (row_out_opt) |buf| alloc.free(buf);
        row_out_opt = alloc.alloc(i8, row_out_len) catch |err| switch (err) {
            error.OutOfMemory => {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS][1x1-stream] row_out scratch {d} bytes failed -> disable streaming\n",
                    .{ row_out_len * @sizeOf(i8) },
                );
                break :streaming;
            },
            else => return err,
        };
        const row_out = row_out_opt.?;

        logDebug("[ZANT][qlinearconv][CMSIS][1x1-stream] row_in={d} row_out={d}\n", .{ row_in_len, row_out_len });

        const zero_adjust_stream: i32 = if (is_u8_input) 128 else 0;
        const zero_restore_stream: i32 = if (is_u8_input) 128 else 0;

        var n_stream: usize = 0;
        while (n_stream < batch_size) : (n_stream += 1) {
            var h_stream: usize = 0;
            while (h_stream < in_height) : (h_stream += 1) {
                // NCHW -> NHWC for a single row
                var c_copy_s: usize = 0;
                while (c_copy_s < in_channels) : (c_copy_s += 1) {
                    var w_copy_s: usize = 0;
                    while (w_copy_s < in_width) : (w_copy_s += 1) {
                        const src_idx_s = ((n_stream * in_channels + c_copy_s) * in_height + h_stream) * in_width + w_copy_s;
                        const dst_idx_s = w_copy_s * in_channels + c_copy_s;
                        const raw_s = @as(i32, @intCast(effective_input.data[src_idx_s]));
                        const centered_s = raw_s - zero_adjust_stream;
                        row_in[dst_idx_s] = @as(i8, @intCast(std.math.clamp(centered_s, -128, 127)));
                    }
                }

                // Run CMSIS for this row
                status_stream = cmsis_nn.conv.arm_convolve_wrapper_s8(
                    &ctx_row,
                    &conv_params_row,
                    &quant_params,
                    &input_dims_row,
                    row_in.ptr,
                    &filter_dims,
                    weight_ptr_s8,
                    &bias_dims,
                    if (bias_ptr) |ptr| @ptrCast(ptr) else null,
                    &output_dims_row,
                    row_out.ptr,
                );
                if (status_stream != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
                    logDebug("[ZANT][qlinearconv][CMSIS][1x1-stream] row conv failed status={d}\n", .{status_stream});
                    return TensorMathError.InvalidDimensions;
                }

                // NHWC row -> NCHW row in output tensor
                var w_write: usize = 0;
                while (w_write < out_width) : (w_write += 1) {
                    var m_write: usize = 0;
                    while (m_write < out_channels) : (m_write += 1) {
                        const src_idx_out = w_write * out_channels + m_write;
                        const v_i32 = @as(i32, @intCast(row_out[src_idx_out]));
                        const adjusted = v_i32 + zero_restore_stream;
                        const dst_idx_out = ((n_stream * out_channels + m_write) * out_height + h_stream) * out_width + w_write;
                        if (is_u8_input) {
                            output.data[dst_idx_out] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
                        } else {
                            output.data[dst_idx_out] = @as(InputType, @intCast(adjusted));
                        }
                    }
                }
            }
        }

        logDebug("[ZANT][qlinearconv][CMSIS][1x1-stream] done\n", .{});
        return;
    }

    // Allocate NHWC s8 buffer for input
    const padded_spatial = in_height * in_width;
    const input_len = batch_size * padded_spatial * in_channels;
    const input_buf = alloc.alloc(i8, input_len) catch |err| switch (err) {
        error.OutOfMemory => {
            logDebug(
                "[ZANT][qlinearconv][CMSIS] input scratch {d} bytes failed -> embedded\n",
                .{ input_len * @sizeOf(i8) },
            );
            return fallbackToEmbedded(
                InputType,
                WeightType,
                ScaleType,
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
        },
        else => return err,
    };
    input_converted = input_buf;
    logDebug("[ZANT][qlinearconv][CMSIS][alloc] input_nhwc_s8={d} bytes\n", .{input_len * @sizeOf(i8)});

    // Fill with input zero-point in s8 domain so padded borders become neutral
    const pad_s8: i8 = @as(i8, @intCast(input_zero_point_s8));
    @memset(input_buf, pad_s8);

    // Copy NCHW -> NHWC without offset (no integrated padding; CMSIS padding handles borders)
    const zero_adjust: i32 = if (is_u8_input) 128 else 0;
    var n_copy: usize = 0;
    while (n_copy < batch_size) : (n_copy += 1) {
        var c_copy: usize = 0;
        while (c_copy < in_channels) : (c_copy += 1) {
            var h_copy: usize = 0;
            while (h_copy < orig_in_height) : (h_copy += 1) {
                var w_copy: usize = 0;
                while (w_copy < orig_in_width) : (w_copy += 1) {
                    const src_idx = ((n_copy * in_channels + c_copy) * orig_in_height + h_copy) * orig_in_width + w_copy;
                    const dst_pixel = ((n_copy * in_height) + h_copy) * in_width + w_copy;
                    const dst_idx = dst_pixel * in_channels + c_copy;
                    const raw = @as(i32, @intCast(effective_input.data[src_idx]));
                    const centered = raw - zero_adjust;
                    input_buf[dst_idx] = @as(i8, @intCast(std.math.clamp(centered, -128, 127)));
                }
            }
        }
    }
    const input_ptr_s8: [*]const i8 = input_buf.ptr;

    // Allocate temporary NHWC s8 buffer for output
    const output_len = output.data.len;
    const output_buf = alloc.alloc(i8, output_len) catch |err| switch (err) {
        error.OutOfMemory => {
            if (group_val == 1 and !is_depthwise) {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS] output scratch {d} bytes failed -> channel chunks\n",
                    .{ output_len * @sizeOf(i8) },
                );
                runCmsisStandardConvChunked(
                    InputType,
                    alloc,
                    output,
                    input_ptr_s8,
                    &conv_params,
                    &input_dims,
                    filter_dims,
                    bias_dims,
                    output_dims,
                    batch_size,
                    out_channels,
                    out_height,
                    out_width,
                    weight_ptr_s8,
                    weight_in_channels,
                    kernel_height,
                    kernel_width,
                    multipliers_buf,
                    shifts_buf,
                    bias_ptr,
                    &ctx,
                    ctx_size_i32,
                    is_u8_input,
                    zero_restore,
                ) catch |chunk_err| switch (chunk_err) {
                    error.OutOfMemory, ChunkError.CmsisKernelFailed => {
                        logDebug(
                            "[ZANT][qlinearconv][CMSIS][chunk] failed err={any} -> embedded fallback\n",
                            .{chunk_err},
                        );
                        try fallbackToEmbedded(
                            InputType,
                            WeightType,
                            ScaleType,
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
                        return;
                    },
                    else => return chunk_err,
                };
                logDebug("[ZANT][qlinearconv][CMSIS][chunk] succeeded\n", .{});
                return;
            }

            logDebug(
                "[ZANT][qlinearconv][CMSIS] output scratch alloc failed (size={d}) -> embedded\n",
                .{ output_len * @sizeOf(i8) },
            );
            try fallbackToEmbedded(
                InputType,
                WeightType,
                ScaleType,
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
            return;
        },
        else => return err,
    };
    output_converted = output_buf;
    const output_ptr_s8: [*]i8 = output_buf.ptr;
    logDebug("[ZANT][qlinearconv][CMSIS][alloc] output_nhwc_s8={d} bytes\n", .{output_len * @sizeOf(i8)});

    if (group_val != 1 and group_val != in_channels) {
        const grouped_input_len = batch_size * in_height * in_width * group_in_channels;
        const grouped_output_len = batch_size * out_height * out_width * group_out_channels;
        grouped_input = alloc.alloc(i8, grouped_input_len) catch |err| switch (err) {
            error.OutOfMemory => {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS] grouped input scratch {d} bytes failed -> embedded\n",
                    .{ grouped_input_len * @sizeOf(i8) },
                );
                return fallbackToEmbedded(
                    InputType,
                    WeightType,
                    ScaleType,
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
            },
            else => return err,
        };
        grouped_output = alloc.alloc(i8, grouped_output_len) catch |err| switch (err) {
            error.OutOfMemory => {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS] grouped output scratch {d} bytes failed -> embedded\n",
                    .{ grouped_output_len * @sizeOf(i8) },
                );
                return fallbackToEmbedded(
                    InputType,
                    WeightType,
                    ScaleType,
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
            },
            else => return err,
        };
        logDebug("[ZANT][qlinearconv][CMSIS][alloc] grouped_in={d} grouped_out={d} bytes\n", .{ grouped_input_len * @sizeOf(i8), grouped_output_len * @sizeOf(i8) });
    }

    // Call CMSIS-NN wrapper (regular or depthwise)
    var status = cmsis_nn.ARM_CMSIS_NN_SUCCESS;
    if (group_val == in_channels and weight_shape[0] == 1) {
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
    } else if (group_val == in_channels and weight_shape[3] == 1 and weight_shape[0] == out_channels) {
        // Depthwise OHWI-variant: weights are [M, kH, kW, 1]. Transpose to [1, kH, kW, M] (small buffer)
        const dw_elems = kernel_height * kernel_width * out_channels;
        const dw_buf = alloc.alloc(i8, dw_elems) catch |err| switch (err) {
            error.OutOfMemory => {
                logDebug(
                    "[ZANT][qlinearconv][CMSIS] depthwise scratch {d} bytes failed -> embedded\n",
                    .{ dw_elems * @sizeOf(i8) },
                );
                return fallbackToEmbedded(
                    InputType,
                    WeightType,
                    ScaleType,
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
            },
            else => return err,
        };
        defer alloc.free(dw_buf);
        logDebug("[ZANT][qlinearconv][CMSIS][alloc] dw_transpose_i8={d} bytes\n", .{dw_elems * @sizeOf(i8)});

        var m: usize = 0;
        while (m < out_channels) : (m += 1) {
            var kh: usize = 0;
            while (kh < kernel_height) : (kh += 1) {
                var kw: usize = 0;
                while (kw < kernel_width) : (kw += 1) {
                    const src_index = ((m * kernel_height + kh) * kernel_width + kw) * 1;
                    const dst_index = (kh * kernel_width + kw) * out_channels + m;
                    dw_buf[dst_index] = @as(i8, @intCast(w.data[src_index]));
                }
            }
        }

        var dw_params = cmsis_nn.DwConvParams{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .ch_mult = 1, // depthwise: ch_mult=1 when group==Cin and Cout==Cin
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(conv_pad_h), .w = @intCast(conv_pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        };
        var dw_filter_dims = cmsis_nn.Dims{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) };
        status = cmsis_nn.conv.arm_depthwise_conv_wrapper_s8(
            &ctx,
            &dw_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &dw_filter_dims,
            @ptrCast(dw_buf.ptr),
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else if (group_val == 1) {
        // Wrapper path for standard conv; allow non-zero scratch except for 1x1
        var wrap_buf_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
        if (wrap_buf_size < 0) wrap_buf_size = 0;
        if (kernel_height == 1 and kernel_width == 1) {
            if (wrap_buf_size != 0) {
                logDebug("[ZANT][qlinearconv][CMSIS] forcing wrapper scratch=0 for 1x1 (was {d})\n", .{wrap_buf_size});
                wrap_buf_size = 0;
            }
        }
        var wrap_ctx = ctx;
        var wrap_dyn: ?[]u8 = null;
        defer if (wrap_dyn) |b| alloc.free(b);
        if (wrap_buf_size > ctx_size_i32) {
            const newb = try alloc.alignedAlloc(u8, @alignOf(i32), @intCast(wrap_buf_size));
            wrap_dyn = newb;
            wrap_ctx = .{ .buf = newb.ptr, .size = @intCast(newb.len) };
        }
        status = cmsis_nn.conv.arm_convolve_wrapper_s8(
            &wrap_ctx,
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
            logDebug("[ZANT][qlinearconv][CMSIS] wrapper failed status={d} -> embedded\n", .{status});
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
                logDebug(
                    "[ZANT][qlinearconv] CMSIS group wrapper failed status={d} attempting direct kernel\n",
                    .{status},
                );
                // Try direct group call
                var direct_buf_size_g = cmsis_nn.conv.arm_convolve_s8_get_buffer_size(&input_group_dims, &filter_group_dims);
                if (direct_buf_size_g < 0) direct_buf_size_g = 0;
                var direct_dyn_g: ?[]u8 = null;
                defer if (direct_dyn_g) |b| alloc.free(b);
                var direct_ctx_g = ctx;
                if (direct_buf_size_g > ctx_size_i32) {
                    const newbg = try alloc.alignedAlloc(u8, @alignOf(i32), @intCast(direct_buf_size_g));
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
        logDebug(
            "[ZANT][qlinearconv] CMSIS kernels failed status={d} -> fallback to embedded\n",
            .{status},
        );
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
    const out_spatial = out_height * out_width;
    const out_batch_stride = out_spatial * out_channels;
    var n_back: usize = 0;
    var src_batch_base_back: usize = 0;
    var dst_batch_base_back: usize = 0;
    while (n_back < batch_size) : (n_back += 1) {
        var pixel: usize = 0;
        while (pixel < out_spatial) : (pixel += 1) {
            var src_idx = src_batch_base_back + pixel * out_channels;
            var dst_idx = dst_batch_base_back + pixel;
            var c: usize = 0;
            while (c < out_channels) : (c += 1) {
                const v_i32 = @as(i32, @intCast(output_buf[src_idx]));
                const adjusted = v_i32 + zero_restore;
                if (is_u8_input) {
                    output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
                } else {
                    output.data[dst_idx] = @as(InputType, @intCast(adjusted));
                }
                src_idx += 1;
                dst_idx += out_spatial;
            }
        }
        src_batch_base_back += out_batch_stride;
        dst_batch_base_back += out_batch_stride;
    }

    logDebug("[ZANT][qlinearconv] CMSIS execution completed successfully\n", .{});
}

pub fn qlinearconv_cmsis_accelerated(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime OutputType: anytype,
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
    return qlinearconv_cmsis_accelerated_impl(
        InputType,
        WeightType,
        ScaleType,
        OutputType,
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
        &pkg_allocator,
    );
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
    var normalized_shape_storage: [4]usize = undefined;
    const effective_weight_shape: []const usize = blk: {
        if (weight_shape.len == 4 and input_shape.len >= 2 and input_shape[1] != 0) {
            const in_channels = input_shape[1];
            const layout = inferWeightLayout(weight_shape, in_channels, null) catch |err| switch (err) {
                TensorMathError.InvalidDimensions => break :blk weight_shape,
                else => return err,
            };

            switch (layout) {
                .standard => break :blk weight_shape,
                .ohwi => {
                    normalized_shape_storage = .{ weight_shape[0], weight_shape[3], weight_shape[1], weight_shape[2] };
                    break :blk normalized_shape_storage[0..];
                },
                .depthwise_khwc => {
                    normalized_shape_storage = .{ weight_shape[0] * weight_shape[3], 1, weight_shape[1], weight_shape[2] };
                    break :blk normalized_shape_storage[0..];
                },
            }
        }
        break :blk weight_shape;
    };

    return conv.calculateOutputShape(T, input_shape, effective_weight_shape, stride, pads, dilations, auto_pad);
}
