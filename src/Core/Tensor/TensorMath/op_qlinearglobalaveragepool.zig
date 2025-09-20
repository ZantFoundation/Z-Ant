const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing global average pool operation for shape calculation
const globalAvgPool = @import("op_globalAveragePool.zig");

/// QLinearGlobalAveragePool operation following ONNX specification
/// Performs quantized global average pooling using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8) of shape (N, C, H, W, ...)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - y: quantized output tensor of shape (N, C, 1, 1, ...)
///
/// Formula: quantized_output = quantize(global_average_pool(dequantize(x)), y_scale, y_zero_point)
pub fn qlinearglobalaveragepool(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
) !Tensor(InputType) {
    // Input validation
    if (x.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_scale.size != 1 or y_scale.size != 1) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_zero_point.size != 1 or y_zero_point.size != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output shape (same as regular GlobalAveragePool)
    const output_shape = try get_qlinearglobalaveragepool_output_shape(x.shape);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized global average pooling
    try lean_qlinearglobalaveragepool(
        InputType,
        ScaleType,
        ZeroPointType,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        &output,
    );

    return output;
}

/// Lean version of QLinearGlobalAveragePool that operates on pre-allocated output tensor
pub fn lean_qlinearglobalaveragepool(
    x: anytype,
    x_scale: anytype,
    x_zero_point: anytype,
    output: anytype,
    y_scale: anytype,
    y_zero_point: anytype,
) !void {
    const x_scale_val = x_scale.data[0];
    const x_zero_point_val = x_zero_point.data[0];
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = y_zero_point.data[0];

    // Handle different tensor dimensions
    const batch_size = if (x.shape.len >= 1) x.shape[0] else 1;
    const channels = if (x.shape.len >= 2) x.shape[1] else 1;

    // Calculate spatial size based on tensor dimensions
    var spatial_size: usize = 1;
    if (x.shape.len >= 3) {
        // For 4D tensors (NCHW format): [N, C, H, W] - spatial dims are H, W
        // For 3D tensors (CHW format): [C, H, W] - spatial dims are H, W
        for (2..x.shape.len) |i| {
            spatial_size *= x.shape[i];
        }
    } else if (x.shape.len == 2) {
        // For 2D tensors: treat as [N, C] - no spatial dimensions
        spatial_size = 1;
    } else if (x.shape.len == 1) {
        // For 1D tensors: treat as [C] - no batch or spatial dimensions
        spatial_size = 1;
    }

    // Process each batch and channel
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            var sum: f64 = 0.0; // Use f64 for better precision in accumulation

            // Sum all spatial elements for this channel
            // Different indexing based on tensor dimensions
            if (x.shape.len == 1) {
                // For 1D tensors: just use the single element
                const input_idx = 0;
                const InputType = @TypeOf(x.data[0]);
                const dequant_val = if (@typeInfo(InputType) == .int)
                    (@as(f32, @floatFromInt(x.data[input_idx])) - @as(f32, @floatFromInt(x_zero_point_val))) * x_scale_val
                else
                    (x.data[input_idx] - @as(InputType, @floatFromInt(x_zero_point_val))) * x_scale_val;
                sum += @as(f64, dequant_val);
            } else {
                // For multi-dimensional tensors
                const channel_start = ((n * channels) + c) * spatial_size;
                for (0..spatial_size) |i| {
                    const input_idx = channel_start + i;
                    // Dequantize each input value and accumulate
                    const InputType = @TypeOf(x.data[0]);
                    const dequant_val = if (@typeInfo(InputType) == .int)
                        (@as(f32, @floatFromInt(x.data[input_idx])) - @as(f32, @floatFromInt(x_zero_point_val))) * x_scale_val
                    else
                        (x.data[input_idx] - @as(InputType, @floatFromInt(x_zero_point_val))) * x_scale_val;
                    sum += @as(f64, dequant_val);
                }
            }

            // Calculate average
            const spatial_size_f32 = @as(f32, @floatFromInt(spatial_size));
            const sum_f32 = @as(f32, @floatCast(sum));
            const avg_float = sum_f32 / spatial_size_f32;

            // Quantize result
            const scaled_result = (avg_float / y_scale_val) + @as(f32, @floatFromInt(y_zero_point_val));

            // Clamp to valid range for output type
            const OutputType = @TypeOf(output.data[0]);
            const min_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.minInt(OutputType)))
            else
                std.math.floatMin(OutputType);
            const max_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
            else
                std.math.floatMax(OutputType);
            const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

            // Store result
            const output_idx = (n * channels) + c;
            output.data[output_idx] = if (@typeInfo(OutputType) == .int)
                @as(OutputType, @intFromFloat(clamped_result))
            else
                @as(OutputType, clamped_result);
        }
    }
}

/// Calculate output shape for QLinearGlobalAveragePool
/// Output shape is (N, C, 1, 1, ...) where all spatial dimensions become 1
pub fn get_qlinearglobalaveragepool_output_shape(input_shape: []const usize) ![]usize {
    if (input_shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);

    // First two dimensions (batch_size, channels) remain the same
    output_shape[0] = input_shape[0]; // N (batch size)
    output_shape[1] = input_shape[1]; // C (channels)

    // All spatial dimensions become 1
    for (2..input_shape.len) |i| {
        output_shape[i] = 1;
    }

    return output_shape;
}
