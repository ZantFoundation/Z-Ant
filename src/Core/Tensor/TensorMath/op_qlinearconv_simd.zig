const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const TensorMathError = @import("../../../Utils/errorHandler.zig").TensorMathError;
const accelerators = @import("../Accelerators/mod.zig");
const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");

const accel_scratch_size = 1024 * 1024;
var accel_scratch: [accel_scratch_size]u8 linksection(".tensor_pool") = undefined;

// Helper function for safe saturation from float to integer output type
inline fn saturate_to_int(comptime OutT: type, value: f32) OutT {
    if (!std.math.isFinite(value)) {
        // For NaN/Inf, return midpoint of range
        const mid = (std.math.minInt(OutT) + std.math.maxInt(OutT)) / 2;
        return @as(OutT, @intCast(mid));
    }

    const rounded = @round(value);
    const min_val = @as(f32, @floatFromInt(std.math.minInt(OutT)));
    const max_val = @as(f32, @floatFromInt(std.math.maxInt(OutT)));

    if (rounded <= min_val) return std.math.minInt(OutT);
    if (rounded >= max_val) return std.math.maxInt(OutT);

    return @as(OutT, @intCast(@as(i64, @intFromFloat(rounded))));
}

// Helper to safely extract scalar value from tensor
inline fn extract_scalar(comptime T: type, tensor: anytype) T {
    if (tensor.data.len == 0) return 0;
    return tensor.data[0];
}

// Helper to safely extract float scale
inline fn extract_scale(tensor: anytype) f32 {
    if (tensor.data.len == 0) return 1.0; // Default scale
    const val = tensor.data[0];
    const scale_f32 = switch (@TypeOf(val)) {
        f32 => val,
        f64 => @as(f32, @floatCast(val)),
        else => @as(f32, @floatFromInt(val)),
    };

    // Ensure positive finite scale
    if (!std.math.isFinite(scale_f32) or scale_f32 <= 0.0) {
        return 1.0; // Fallback to safe value
    }
    return scale_f32;
}

// Helper to get per-channel scale value
inline fn get_channel_scale(tensor: anytype, channel: usize) f32 {
    if (tensor.data.len == 0) return 1.0;

    // Per-tensor (scalar) vs per-channel
    const idx = if (tensor.data.len == 1) 0 else @min(channel, tensor.data.len - 1);
    const val = tensor.data[idx];

    const scale_f32 = switch (@TypeOf(val)) {
        f32 => val,
        f64 => @as(f32, @floatCast(val)),
        else => @as(f32, @floatFromInt(val)),
    };

    if (!std.math.isFinite(scale_f32) or scale_f32 <= 0.0) {
        return 1.0;
    }
    return scale_f32;
}

// Helper to get per-channel zero point
inline fn get_channel_zero_point(tensor: anytype, channel: usize) i32 {
    if (tensor.data.len == 0) return 0;

    // Per-tensor (scalar) vs per-channel
    const idx = if (tensor.data.len == 1) 0 else @min(channel, tensor.data.len - 1);
    return @as(i32, @intCast(tensor.data[idx]));
}

// CMSIS-NN extern definitions are centralized in the STM32N6 accelerator
// module so they can be reused by future Helium-accelerated operators.
const cmsis_nn_dims = cmsis_nn.Dims;
const cmsis_nn_context = cmsis_nn.Context;
const cmsis_nn_conv_params = cmsis_nn.ConvParams;
const cmsis_nn_per_channel_quant_params = cmsis_nn.PerChannelQuantParams;
const cmsis_nn_functions = cmsis_nn.functions;

fn tryDirectCmsisNnQLinearConv(
    comptime InputType: type,
    comptime WeightType: type,
    comptime OutputType: type,
    comptime BiasType: type,
    x: *const Tensor(InputType),
    x_scale_f: f32,
    x_zp: i32,
    w: *const Tensor(WeightType),
    w_scale: anytype,
    w_zero_point: anytype,
    output: *Tensor(OutputType),
    y_scale_f: f32,
    y_zp: i32,
    bias: ?*const Tensor(BiasType),
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    auto_pad: []const u8,
) bool {
    // Only proceed when STM32N6 acceleration is enabled (where CMSIS-NN sources will be linked)
    if (!accelerators.canUseCmsisHelium()) return false;
    if (!cmsis_nn.supportsConvolveS8()) return false;

    _ = auto_pad;
    _ = w_zero_point; // TODO: implement per-channel weight zero points
    _ = pad_bottom; // TODO: implement asymmetric padding
    _ = pad_right; // TODO: implement asymmetric padding

    // Only handle simple cases for now
    if (groups != 1) return false;
    if (dilation_h != 1 or dilation_w != 1) return false;
    if (InputType != u8 or WeightType != i8 or OutputType != u8) return false;
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) return false;

    var scratch = std.heap.FixedBufferAllocator.init(&accel_scratch);
    defer scratch.reset();
    var alloc_inst = scratch.allocator();
    const alloc = &alloc_inst;

    // Extract dimensions
    const batch_size = @as(i32, @intCast(x.shape[0]));
    const in_height = @as(i32, @intCast(x.shape[2]));
    const in_width = @as(i32, @intCast(x.shape[3]));
    const in_channels = @as(i32, @intCast(x.shape[1]));

    const out_channels = @as(i32, @intCast(w.shape[0]));
    const kernel_height = @as(i32, @intCast(w.shape[2]));
    const kernel_width = @as(i32, @intCast(w.shape[3]));

    const out_height = @as(i32, @intCast(output.shape[2]));
    const out_width = @as(i32, @intCast(output.shape[3]));

    // Setup CMSIS-NN dimensions
    const input_dims = cmsis_nn_dims{
        .n = batch_size,
        .h = in_height,
        .w = in_width,
        .c = in_channels,
    };

    const filter_dims = cmsis_nn_dims{
        .n = out_channels,
        .h = kernel_height,
        .w = kernel_width,
        .c = in_channels,
    };

    const output_dims = cmsis_nn_dims{
        .n = batch_size,
        .h = out_height,
        .w = out_width,
        .c = out_channels,
    };

    const bias_dims = cmsis_nn_dims{
        .n = 1,
        .h = 1,
        .w = 1,
        .c = out_channels,
    };

    // Get buffer size and allocate
    const buffer_size = cmsis_nn_functions.arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    if (buffer_size <= 0) return false;

    const buffer = alloc.alignedAlloc(u8, @alignOf(i16), @as(usize, @intCast(buffer_size))) catch return false;
    defer alloc.free(buffer);

    // Setup context
    const ctx = cmsis_nn_context{
        .buf = buffer.ptr,
        .size = buffer_size,
    };

    // Setup convolution parameters
    const conv_params = cmsis_nn_conv_params{
        .input_offset = -x_zp,
        .output_offset = y_zp,
        .stride = .{ .h = @as(i32, @intCast(stride_h)), .w = @as(i32, @intCast(stride_w)) },
        .padding = .{ .h = @as(i32, @intCast(pad_top)), .w = @as(i32, @intCast(pad_left)) },
        .dilation = .{ .h = @as(i32, @intCast(dilation_h)), .w = @as(i32, @intCast(dilation_w)) },
        .activation = .{ .min = -128, .max = 127 },
    };

    // Setup quantization parameters (simplified - per-tensor)
    const w_scale_0 = get_channel_scale(w_scale, 0);
    const combined_scale = x_scale_f * w_scale_0 / y_scale_f;

    // Convert to CMSIS-NN fixed-point format
    const scale_factor: f32 = 1073741824.0; // 2^30
    var multiplier = @as(i32, @intFromFloat(combined_scale * scale_factor));
    var shift: i32 = 30;

    // Normalize multiplier to be in [0.5, 1.0) range
    while (multiplier >= 1073741824) { // 2^30
        multiplier >>= 1;
        shift -= 1;
    }

    // Need to create arrays for per-channel params
    var multiplier_array = [_]i32{multiplier};
    var shift_array = [_]i32{shift};

    const quant_params = cmsis_nn_per_channel_quant_params{
        .multiplier = &multiplier_array,
        .shift = &shift_array,
    };

    // Prepare bias data
    var bias_data: ?[*]const i32 = null;
    var bias_storage: []i32 = undefined;
    var bias_allocated = false;

    if (bias) |bias_tensor| {
        bias_storage = alloc.alloc(i32, @as(usize, @intCast(out_channels))) catch return false;
        bias_allocated = true;

        for (0..@as(usize, @intCast(out_channels))) |i| {
            const bias_val = if (i < bias_tensor.data.len) bias_tensor.data[i] else 0;
            // Convert bias to CMSIS-NN format
            bias_storage[i] = @as(i32, @intCast(bias_val));
        }
        bias_data = bias_storage.ptr;
    }
    defer if (bias_allocated) alloc.free(bias_storage);

    // Call CMSIS-NN directly with quantized data!
    const result = cmsis_nn_functions.arm_convolve_s8(
        &ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        @as([*]const i8, @ptrCast(x.data.ptr)),
        &filter_dims,
        @as([*]const i8, @ptrCast(w.data.ptr)),
        &bias_dims,
        bias_data,
        null, // upscale_dims
        &output_dims,
        @as([*]i8, @ptrCast(output.data.ptr)),
    );

    return result == cmsis_nn.ARM_CMSIS_NN_SUCCESS;
}

fn tryAcceleratedQLinearConv(
    comptime InputType: type,
    comptime WeightType: type,
    comptime OutputType: type,
    comptime BiasType: type,
    x: *const Tensor(InputType),
    x_scale_f: f32,
    x_zp: i32,
    w: *const Tensor(WeightType),
    w_scale: anytype,
    w_zero_point: anytype,
    output: *Tensor(OutputType),
    y_scale_f: f32,
    y_zp: i32,
    bias: ?*const Tensor(BiasType),
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    auto_pad: []const u8,
) bool {
    if (!accelerators.canUseCmsisHelium()) return false;

    // Try direct CMSIS-NN quantized convolution first
    if (tryDirectCmsisNnQLinearConv(InputType, WeightType, OutputType, BiasType, x, x_scale_f, x_zp, w, w_scale, w_zero_point, output, y_scale_f, y_zp, bias, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right, dilation_h, dilation_w, groups, auto_pad)) {
        return true;
    }

    // Fallback to f32 path (the old double-quantization path)
    // This should not be reached with proper CMSIS-NN integration
    return false;
}

/// QLinearConv implementation following ONNX v10 specification exactly
/// https://onnx.ai/onnx/operators/onnx__QLinearConv.html
///
/// Formula: Y = (X_dequant * W_dequant + B_dequant) quantized to output
/// Where:
/// - X_dequant = (X - x_zero_point) * x_scale
/// - W_dequant = (W - w_zero_point) * w_scale
/// - B_dequant = B * (x_scale * w_scale)  [bias already quantized with this scale]
/// - Y_quant = Y_dequant / y_scale + y_zero_point
pub fn qlinearconv_onnx_v10(
    comptime InputType: type, // T1: int8 or uint8
    comptime WeightType: type, // T2: int8 or uint8
    comptime ScaleType: type, // tensor(float)
    comptime OutputType: type, // T3: int8 or uint8
    comptime BiasType: type, // T4: int32
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype, // Flexible to handle different zero point tensor types
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype, // Flexible to handle different zero point tensor types
    output: *Tensor(OutputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype, // Flexible to handle different zero point tensor types
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    // Validate input shapes per ONNX spec
    if (x.shape.len < 3 or w.shape.len < 3) return error.InvalidShape1;

    // Extract dimensions: x=[N, C, H, W], w=[M, C/group, kH, kW]
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    const in_height = x.shape[2];
    const in_width = if (x.shape.len > 3) x.shape[3] else 1;

    const out_channels = w.shape[0];
    const kernel_height = w.shape[2];
    const kernel_width = if (w.shape.len > 3) w.shape[3] else 1;

    // Get parameters with defaults
    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;

    const pad_top = if (pads) |p| p[0] else 0;
    const pad_left = if (pads) |p| (if (p.len > 1) p[1] else p[0]) else 0;
    const pad_bottom = if (pads) |p| (if (p.len > 2) p[2] else p[0]) else 0;
    const pad_right = if (pads) |p| (if (p.len > 3) p[3] else (if (p.len > 1) p[1] else p[0])) else 0;

    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    const groups = group orelse 1;

    // Calculate output dimensions
    const out_height = (in_height + pad_top + pad_bottom - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    const out_width = (in_width + pad_left + pad_right - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    // Validate output tensor shape
    if (output.shape.len < 3 or
        output.shape[0] != batch_size or
        output.shape[1] != out_channels or
        output.shape[2] != out_height or
        (output.shape.len > 3 and output.shape[3] != out_width))
    {
        return error.InvalidShape2;
    }

    // Extract quantization parameters per ONNX spec
    const x_scale_f = extract_scale(x_scale);
    const x_zp = if (x_zero_point.data.len > 0) @as(i32, @intCast(x_zero_point.data[0])) else 0;
    const y_scale_f = extract_scale(y_scale);
    const y_zp = if (y_zero_point.data.len > 0) @as(i32, @intCast(y_zero_point.data[0])) else 0;

    // std.debug.print("QLinearConv ONNX v10: x_scale={d:.6}, x_zp={}, y_scale={d:.6}, y_zp={}\n", .{ x_scale_f, x_zp, y_scale_f, y_zp });
    // std.debug.print("Shapes: x={any}, w={any}, out={any}\n", .{ x.shape, w.shape, output.shape });
    // std.debug.print("Conv params: stride=[{},{}], pad=[{},{},{},{}], groups={}\n", .{ stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right, groups });

    // Validation check
    if (x_scale_f <= 0.0 or y_scale_f <= 0.0) {
        //std.debug.print("ERROR: Invalid scales x_scale={d:.6}, y_scale={d:.6}\n", .{ x_scale_f, y_scale_f });
        return error.InvalidShape3;
    }

    if (tryAcceleratedQLinearConv(
        InputType,
        WeightType,
        OutputType,
        BiasType,
        x,
        x_scale_f,
        x_zp,
        w,
        w_scale,
        w_zero_point,
        output,
        y_scale_f,
        y_zp,
        bias,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        dilation_h,
        dilation_w,
        groups,
        auto_pad,
    )) return;

    // Detailed analysis of quantization parameters
    //const eff = (x_scale_f * get_channel_scale(w_scale, 0)) / y_scale_f;
    //std.debug.print("eff=(sx*sw)/sy ~ {d}\n", .{eff});
    // Se eff >> 1, aspettati saturazioni: y ≈ acc_int * eff + y_zp.
    //std.debug.print("types: X={s} W={s} Y={s}\n", .{ @typeName(InputType), @typeName(WeightType), @typeName(OutputType) });
    //std.debug.print("x_zp={}, y_zp={}, w_zp[0..?]=...\n", .{
    //    if (x_zero_point.data.len > 0) @as(i32, @intCast(x_zero_point.data[0])) else 0,
    //    if (y_zero_point.data.len > 0) @as(i32, @intCast(y_zero_point.data[0])) else 0,
    //});
    //std.debug.print("len(w_scale)={}, out_channels={}\n", .{ w_scale.data.len, out_channels });

    //std.debug.print("VALIDATION OK, starting convolution\n", .{});

    // Main convolution loop
    for (0..batch_size) |n| {
        for (0..out_channels) |m| {
            // Get per-channel weight quantization parameters
            const w_scale_m = get_channel_scale(w_scale, m);
            const w_zp_m = get_channel_zero_point(w_zero_point, m);

            for (0..out_height) |oh| {
                for (0..out_width) |ow| {

                    // Accumulate in float following ONNX spec mathematical operations
                    var acc_float: f32 = 0.0;

                    // Input channels for this output channel (considering groups)
                    const group_size = in_channels / groups;
                    const group_idx = m / (out_channels / groups);
                    const c_start = group_idx * group_size;
                    const c_end = c_start + group_size;

                    for (c_start..c_end) |c| {
                        for (0..kernel_height) |kh| {
                            for (0..kernel_width) |kw| {
                                // Input coordinates with padding and dilation
                                const ih = @as(i64, @intCast(oh * stride_h + kh * dilation_h)) - @as(i64, @intCast(pad_top));
                                const iw = @as(i64, @intCast(ow * stride_w + kw * dilation_w)) - @as(i64, @intCast(pad_left));

                                // Check bounds (zero padding outside)
                                if (ih >= 0 and ih < in_height and iw >= 0 and iw < in_width) {
                                    const x_idx = ((n * in_channels + c) * in_height + @as(usize, @intCast(ih))) * in_width + @as(usize, @intCast(iw));
                                    const w_idx = ((m * (in_channels / groups) + (c - c_start)) * kernel_height + kh) * kernel_width + kw;

                                    // ONNX QLinearConv math: dequantize then multiply
                                    const x_val = @as(f32, @floatFromInt(@as(i32, @intCast(x.data[x_idx])) - @as(i32, @intCast(x_zp))));
                                    const w_val = @as(f32, @floatFromInt(@as(i32, @intCast(w.data[w_idx])) - w_zp_m));

                                    // Multiply dequantized values
                                    acc_float += (x_val * x_scale_f) * (w_val * w_scale_m);
                                }
                            }
                        }
                    }

                    // Add bias if present (bias already quantized with x_scale * w_scale per spec)
                    if (bias) |b| {
                        if (m < b.data.len) {
                            const bias_val = b.data[m];
                            const bias_f32 = switch (@TypeOf(bias_val)) {
                                f32 => bias_val,
                                f64 => @as(f32, @floatCast(bias_val)),
                                else => @as(f32, @floatFromInt(bias_val)),
                            };
                            const bias_dequant = bias_f32 * (x_scale_f * w_scale_m);
                            acc_float += bias_dequant;
                        }
                    }

                    // Quantize to output: Y = round(Y_float / y_scale + y_zero_point)
                    const y_float = acc_float / y_scale_f + @as(f32, @floatFromInt(y_zp));

                    // Store result with saturation
                    const out_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                    output.data[out_idx] = saturate_to_int(OutputType, y_float);
                }
            }
        }
    }

    // Output statistics for debugging
    var min_val: i32 = std.math.maxInt(i32);
    var max_val: i32 = std.math.minInt(i32);
    var nonzero_count: usize = 0;

    for (output.data) |val| {
        const val_i32 = @as(i32, @intCast(val));
        if (val_i32 < min_val) min_val = val_i32;
        if (val_i32 > max_val) max_val = val_i32;
        if (val != 0) nonzero_count += 1;
    }

    //std.debug.print("Output stats: min={}, max={}, nonzero={}/{}\n", .{ min_val, max_val, nonzero_count, output.data.len });
    //std.debug.print("QLinearConv ONNX v10 COMPLETED SUCCESSFULLY\n", .{});
}

// Compatibility wrapper maintaining original API
pub fn qlinearconv_simd_lean(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime OutputType: anytype, // <-- usa davvero l'OutputType
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype, // Flexible type to handle different zero point types
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype, // Flexible type to handle different zero point types
    output: *Tensor(OutputType), // <-- qui
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype, // Flexible type to handle different zero point types
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    return qlinearconv_onnx_v10(InputType, WeightType, ScaleType, OutputType, BiasType, x, x_scale, x_zero_point, w, w_scale, w_zero_point, output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad);
}
