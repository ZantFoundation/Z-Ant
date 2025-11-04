const zant = @import("../../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const c = @cImport({
    @cInclude("arm_nnfunctions.h");
});
const std = @import("std");
const pkg_allocator = zant.utils.allocator.allocator;
const quantize = @import("../../QuantTensorMath/op_quantize.zig");
const dequantize = @import("../../QuantTensorMath/op_dequantize.zig");

pub fn tryConv(
    comptime T: type,
    input: *const Tensor(T),
    weight: *const Tensor(T),
    output: *Tensor(T),
    bias: ?[]const T,
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
) !bool {

    // ========== VALIDATE INPUTS ==========
    if (stride == null or pads == null or dilations == null) {
        return error.InvalidParameters;
    }

    // ========== PREPARE DIMENSIONS ==========
    const input_dims: c.cmsis_nn_dims = .{
        .n = @intCast(input.shape[0]),
        .h = @intCast(input.shape[2]),
        .w = @intCast(input.shape[3]),
        .c = @intCast(input.shape[1]),
    };

    const filter_dims: c.cmsis_nn_dims = .{
        .n = @intCast(weight.shape[0]), // Number of output channels
        .h = @intCast(weight.shape[2]),
        .w = @intCast(weight.shape[3]),
        .c = @intCast(weight.shape[1]), // Number of input channels
    };

    const output_dims: c.cmsis_nn_dims = .{
        .n = @intCast(output.shape[0]),
        .h = @intCast(output.shape[2]),
        .w = @intCast(output.shape[3]),
        .c = @intCast(output.shape[1]),
    };

    const num_output_channels = weight.shape[0];

    const bias_dims: c.cmsis_nn_dims = .{
        .n = @intCast(num_output_channels),
        .h = 1,
        .w = 1,
        .c = 1,
    };

    // ========== QUANTIZE INPUT (per-tensor) ==========
    const input_quant = try quantize.quantize_struct(T, i8, input, quantize.quantScheme.ASYM);
    defer input_quant.tensor.deinit();

    const input_scale = input_quant.scale;
    const input_zero = input_quant.zero;
    const input_data: [*c]const i8 = @ptrCast(input_quant.tensor.data.ptr);

    // ========== QUANTIZE WEIGHT (per-channel) ==========
    const weight_quant = try quantize.quantize_per_channel(T, i8, weight, quantize.quantScheme.ASYM);
    defer weight_quant.tensor.deinit();
    defer pkg_allocator.free(weight_quant.scales);
    defer pkg_allocator.free(weight_quant.zeros);

    const filter_data: [*c]const i8 = @ptrCast(weight_quant.tensor.data.ptr);

    // ========== BUILD PER-CHANNEL QUANTIZATION PARAMETERS ==========
    // Allocate arrays for multiplier and shift (required by CMSIS-NN)
    var multipliers = try pkg_allocator.alloc(i32, num_output_channels);
    defer pkg_allocator.free(multipliers);

    var shifts = try pkg_allocator.alloc(i32, num_output_channels);
    defer pkg_allocator.free(shifts);

    // Calculate multiplier and shift for each output channel
    for (0..num_output_channels) |ch| {
        // Combined scale: input_scale × weight_scale[ch]
        const combined_scale = input_scale * weight_quant.scales[ch];

        // Convert float scale to fixed-point (multiplier, shift)
        const quant_params_ch = try scaleToMultiplierAndShift(combined_scale);

        multipliers[ch] = quant_params_ch.multiplier;
        shifts[ch] = quant_params_ch.shift;
    }

    // Build CMSIS-NN per-channel quantization parameters
    const quant_params: c.cmsis_nn_per_channel_quant_params = .{
        .multiplier = @ptrCast(multipliers.ptr),
        .shift = @ptrCast(shifts.ptr),
    };

    // ========== QUANTIZE BIAS (per-channel) ==========
    var bias_buffer = try pkg_allocator.alloc(i32, num_output_channels);
    defer pkg_allocator.free(bias_buffer);

    if (bias) |b| {
        // Validate bias length
        if (b.len != num_output_channels) {
            return error.BiasLengthMismatch;
        }

        // Quantize bias: bias_i32[ch] = round(bias_f32[ch] / (input_scale × weight_scale[ch]))
        for (b, 0..) |bias_val, ch| {
            const bias_quant_scale = input_scale * weight_quant.scales[ch];

            if (bias_quant_scale == 0.0) {
                bias_buffer[ch] = 0;
            } else {
                bias_buffer[ch] = @intFromFloat(@round(bias_val / bias_quant_scale));
            }
        }
    } else {
        // No bias provided, set all to zero
        @memset(bias_buffer, 0);
    }

    const bias_data: [*c]const i32 = @ptrCast(bias_buffer.ptr);

    // ========== PREPARE CONVOLUTION PARAMETERS ==========
    const conv_params: c.cmsis_nn_conv_params = .{
        .input_offset = -input_zero,
        .output_offset = 0,
        .activation = .{
            .min = -128,
            .max = 127,
        },
        .stride = .{
            .h = @intCast(stride.?[0]),
            .w = @intCast(stride.?[1]),
        },
        .padding = .{
            .h = @intCast(pads.?[0]),
            .w = @intCast(pads.?[1]),
        },
        .dilation = .{
            .h = @intCast(dilations.?[0]),
            .w = @intCast(dilations.?[1]),
        },
    };

    // ========== ALLOCATE CMSIS BUFFER ==========
    const buffer_size = c.arm_convolve_wrapper_s8_get_buffer_size(
        &conv_params,
        &input_dims,
        &filter_dims,
        &output_dims,
    );

    const buf = try pkg_allocator.alloc(u8, @intCast(buffer_size));
    defer pkg_allocator.free(buf);

    const ctx: c.cmsis_nn_context = .{
        .buf = buf.ptr,
        .size = buffer_size,
    };

    // ========== ALLOCATE OUTPUT BUFFER (quantized) ==========
    var output_quantized = try Tensor(i8).fromShape(&pkg_allocator, output.shape);
    defer output_quantized.deinit();
    const output_data: [*c]i8 = @ptrCast(output_quantized.data.ptr);

    // ========== CALL CMSIS-NN CONVOLUTION ==========
    const status = c.arm_convolve_wrapper_s8(
        &ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        input_data,
        &filter_dims,
        filter_data,
        &bias_dims,
        bias_data,
        &output_dims,
        output_data,
    );

    // ========== CHECK RESULT ==========
    if (status != c.ARM_CMSIS_NN_SUCCESS) {
        std.debug.print("CMSIS-NN convolution failed with status: {d}\n", .{status});
        return false;
    }

    // ========== DEQUANTIZE OUTPUT ==========

    // Dequantize output
    var output_dequantized = try dequantize.dequantize(i8, T, &output_quantized);
    defer output_dequantized.deinit();

    // Copy results to output tensor
    @memcpy(output.data, output_dequantized.data);

    // // Calculate average output scale (weighted by channel scales)
    // var output_scale: f32 = 0.0;
    // for (0..num_output_channels) |ch| {
    //     output_scale += input_scale * weight_quant.scales[ch];
    // }
    // output_scale /= @as(f32, @floatFromInt(num_output_channels));

    // // Dequantize from i8 to T
    // try dequantizeInPlace(i8, T, output_quantized.data, output.data, output_scale, 0);

    return true;
}

// ========== HELPER FUNCTIONS ==========

/// Converts a floating-point scale to fixed-point (multiplier, shift) format
/// required by CMSIS-NN per-channel quantization.
///
/// The formula is: effective_scale = multiplier × 2^(-shift)
/// where multiplier is a Q31 fixed-point number
///
/// Parameters:
/// - scale: floating-point scale factor to convert
///
/// Returns: struct containing:
/// - multiplier: Q31 fixed-point multiplier (range: [-2^30, 2^30-1])
/// - shift: bit shift amount (range: [-31, 31])
fn scaleToMultiplierAndShift(scale: f32) !struct { multiplier: i32, shift: i32 } {
    if (scale <= 0.0) {
        return .{ .multiplier = 0, .shift = 0 };
    }

    // Find the optimal exponent (shift)
    // We want: scale = multiplier × 2^(-shift)
    // So: multiplier = scale × 2^shift
    // We need multiplier in range [-2^30, 2^30-1] for Q31 representation
    const log2_scale = @log2(scale);
    var shift: i32 = -@as(i32, @intFromFloat(@ceil(log2_scale)));

    // Clamp shift to valid range [-31, 31]
    shift = @max(-31, @min(31, shift));

    // Calculate multiplier as Q31 fixed-point
    // Q31 format: multiply by 2^31 to convert to integer
    const shifted_scale = scale * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(shift)));
    var multiplier: i32 = @intFromFloat(@round(shifted_scale * 2147483648.0)); // 2^31

    // Clamp multiplier to Q30 range (CMSIS-NN uses Q30, not full Q31)
    // Range: [-(2^30), 2^30-1]
    multiplier = @max(-(1 << 30), @min((1 << 30) - 1, multiplier));

    return .{ .multiplier = multiplier, .shift = shift };
}

/// Dequantizes data in-place from quantized integer to floating-point
///
/// Formula: output[i] = (input[i] - zero_point) × scale
fn dequantizeInPlace(
    comptime InputType: type,
    comptime OutputType: type,
    input_data: []const InputType,
    output_data: []OutputType,
    scale: f32,
    zero_point: i32,
) !void {
    if (input_data.len != output_data.len) {
        return error.SizeMismatch;
    }

    for (input_data, 0..) |val, i| {
        const dequant_val = (@as(f32, @floatFromInt(val)) - @as(f32, @floatFromInt(zero_point))) * scale;
        output_data[i] = @floatCast(dequant_val);
    }
}
