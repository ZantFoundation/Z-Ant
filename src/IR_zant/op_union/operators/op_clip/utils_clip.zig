const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const DTypeValue = Uops.DTypeValue;
const Any = Uops.Any;

/// Returns the shape of the output tensor for the clip operation.
/// For clip operation, the output shape is identical to the input shape.
pub fn get_clip_output_shape(comptime T: type, inputTensor: *const Tensor(T), minTensor: ?*const Tensor(T), maxTensor: ?*const Tensor(T)) ![]usize {
    const allocator = inputTensor.allocator;
    _ = minTensor;
    _ = maxTensor;
    const shape = try allocator.alloc(usize, inputTensor.shape.len);
    @memcpy(shape, inputTensor.shape);
    return shape;
}

/// https://onnx.ai/onnx/operators/onnx__Clip.html
pub fn lowerClip(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    out_shape: []const usize,
    strideA: []const isize,
    out_dtype: DType, // promoted element type
    min: DTypeValue,
    max: DTypeValue,
) usize { // returns id of result buffer

    // -- Set-up phase --
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // -- Flat element loop --
    var nelem: usize = 1;
    for (out_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

    const id_tanh = b.push(.CLIP, out_dtype, &.{id_loadA}, Any{ .clip_bounds = .{ .type = out_dtype, .min = min, .max = max } });

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_tanh }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}

/// Clips quantized tensor elements element-wise in the quantized domain.
/// This avoids the need for dequantization/quantization round-trips.
pub fn clip_quantized_lean(
    comptime InputType: type,
    inputTensor: *const Tensor(InputType),
    input_scale: f32,
    input_zero_point: InputType,
    min_val: f32,
    max_val: f32,
    outputTensor: *Tensor(InputType),
    output_scale: f32,
    output_zero_point: InputType,
) !void {
    const input_zp_f: f32 = @floatFromInt(input_zero_point);
    const output_zp_f: f32 = @floatFromInt(output_zero_point);

    // Calculate quantized bounds for input domain
    const quantized_min_f = min_val / input_scale + input_zp_f;
    const quantized_max_f = max_val / input_scale + input_zp_f;

    // Clamp to valid range for the input type
    const type_min: f32 = @floatFromInt(std.math.minInt(InputType));
    const type_max: f32 = @floatFromInt(std.math.maxInt(InputType));

    const quantized_min_clamped = @max(quantized_min_f, type_min);
    const quantized_max_clamped = @min(quantized_max_f, type_max);

    const quantized_min: InputType = @intFromFloat(@round(quantized_min_clamped));
    const quantized_max: InputType = @intFromFloat(@round(quantized_max_clamped));

    // If input and output have same scale/zero_point, we can clip directly
    const same_quantization = @abs(input_scale - output_scale) < 1e-6 and input_zero_point == output_zero_point;

    if (same_quantization) {
        // Direct clipping in quantized domain
        var i: usize = 0;
        const chunk_size = 32;

        while (i + chunk_size <= inputTensor.size) : (i += chunk_size) {
            comptime var j = 0;
            inline while (j < chunk_size) : (j += 1) {
                outputTensor.data[i + j] = @min(@max(inputTensor.data[i + j], quantized_min), quantized_max);
            }
        }

        while (i < inputTensor.size) : (i += 1) {
            outputTensor.data[i] = @min(@max(inputTensor.data[i], quantized_min), quantized_max);
        }
    } else {
        // Need to requantize: input_domain -> real -> output_domain
        var i: usize = 0;
        const chunk_size = 16; // Smaller chunks due to more computation

        while (i + chunk_size <= inputTensor.size) : (i += chunk_size) {
            comptime var j = 0;
            inline while (j < chunk_size) : (j += 1) {
                // Clip in quantized input domain first
                const clipped_input = @min(@max(inputTensor.data[i + j], quantized_min), quantized_max);

                // Convert to real value
                const real_val = (@as(f32, @floatFromInt(clipped_input)) - input_zp_f) * input_scale;

                // Quantize to output domain
                const output_quantized_f = real_val / output_scale + output_zp_f;
                const output_quantized_clamped = @max(@min(output_quantized_f, type_max), type_min);
                outputTensor.data[i + j] = @intFromFloat(@round(output_quantized_clamped));
            }
        }

        while (i < inputTensor.size) : (i += 1) {
            const clipped_input = @min(@max(inputTensor.data[i], quantized_min), quantized_max);
            const real_val = (@as(f32, @floatFromInt(clipped_input)) - input_zp_f) * input_scale;
            const output_quantized_f = real_val / output_scale + output_zp_f;
            const output_quantized_clamped = @max(@min(output_quantized_f, type_max), type_min);
            outputTensor.data[i] = @intFromFloat(@round(output_quantized_clamped));
        }
    }
}
