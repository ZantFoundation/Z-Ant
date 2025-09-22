const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
// helper used to convert quantized values to floating point
fn toF32(comptime T: type, value: T) f32 {
    return switch (@typeInfo(T)) {
        .Float => @as(f32, @floatCast(value)),
        .ComptimeFloat => @floatCast(f32, value),
        .Int => @as(f32, @floatFromInt(value)),
        .ComptimeInt => @as(f32, @floatFromInt(value)),
        else => @compileError("QLinearGlobalAveragePool supports only numeric element types"),
    };
}

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
/// - y: floating point output tensor of shape (N, C, 1, 1, ...)
///
/// Formula: output = global_average_pool(dequantize(x))
pub fn qlinearglobalaveragepool(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
) !Tensor(f32) {
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
    var output = try Tensor(f32).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized global average pooling
    try lean_qlinearglobalaveragepool(
        x,
        x_scale,
        x_zero_point,
        &output,
        y_scale,
        y_zero_point,
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
    comptime {
        const output_info = @typeInfo(@TypeOf(output));
        if (output_info != .Pointer or output_info.Pointer.child != Tensor(f32)) {
            @compileError("QLinearGlobalAveragePool lean implementation requires a Tensor(f32) output");
        }
    }

    const out_tensor: *Tensor(f32) = output;

    if (x.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    if (out_tensor.shape.len != x.shape.len) {
        return TensorMathError.ShapeMismatch;
    }

    if (out_tensor.shape[0] != x.shape[0] or out_tensor.shape[1] != x.shape[1]) {
        return TensorMathError.ShapeMismatch;
    }

    for (2..out_tensor.shape.len) |i| {
        if (out_tensor.shape[i] != 1) {
            return TensorMathError.ShapeMismatch;
        }
    }

    const x_scale_val = toF32(@TypeOf(x_scale.data[0]), x_scale.data[0]);
    const x_zero_point_val = toF32(@TypeOf(x_zero_point.data[0]), x_zero_point.data[0]);
    const _ = toF32(@TypeOf(y_scale.data[0]), y_scale.data[0]);
    const _ = toF32(@TypeOf(y_zero_point.data[0]), y_zero_point.data[0]);

    // Handle different tensor dimensions
    const batch_size = if (x.shape.len >= 1) x.shape[0] else 1;
    const channels = if (x.shape.len >= 2) x.shape[1] else 1;

    // Calculate spatial size based on tensor dimensions
    var spatial_size: usize = 1;
    for (2..x.shape.len) |i| {
        spatial_size *= x.shape[i];
    }

    if (spatial_size == 0) spatial_size = 1;

    // Process each batch and channel
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            var sum: f64 = 0.0; // Use f64 for better precision in accumulation

            // Sum all spatial elements for this channel
            // Different indexing based on tensor dimensions
            const channel_start = ((n * channels) + c) * spatial_size;
            for (0..spatial_size) |i| {
                const input_idx = channel_start + i;
                const InputElemType = @TypeOf(x.data[0]);
                const input_val = toF32(InputElemType, x.data[input_idx]);
                const dequant_val = (input_val - x_zero_point_val) * x_scale_val;
                sum += @as(f64, @floatCast(dequant_val));
            }

            // Calculate average
            const spatial_size_f32 = @as(f32, @floatFromInt(spatial_size));
            const sum_f32 = @as(f32, @floatCast(sum));
            const avg_float = sum_f32 / spatial_size_f32;

            const output_idx = (n * channels) + c;
            out_tensor.data[output_idx] = avg_float;
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
