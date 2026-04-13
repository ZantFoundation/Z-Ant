const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

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
