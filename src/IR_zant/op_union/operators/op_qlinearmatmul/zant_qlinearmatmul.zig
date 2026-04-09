const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

// Import existing matmul operation for shape calculation
const matmul = @import("../op_matMul/zant_matMul.zig");

const utils = @import("utils_qlinearmatmul.zig");

/// QLinearMatMul operation following ONNX specification
/// Performs quantized matrix multiplication using linear quantization scheme
///
/// INPUTS:
/// - a: quantized input tensor A (typically int8/uint8)
/// - a_scale: scale factor for input A quantization
/// - a_zero_point: zero point for input A quantization
/// - b: quantized input tensor B
/// - b_scale: scale factor for input B quantization
/// - b_zero_point: zero point for input B quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(matmul(dequantize(a), dequantize(b)), y_scale, y_zero_point)
pub fn qlinear_mat_mul(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    a: *const Tensor(InputType),
    a_scale: *const Tensor(ScaleType),
    a_zero_point: *const Tensor(ZeroPointType),
    b: *const Tensor(InputType),
    b_scale: *const Tensor(ScaleType),
    b_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
) !Tensor(InputType) {
    // Input validation
    if (a.shape.len != b.shape.len) {
        return error.InputTensorDifferentShape;
    }
    if (a_scale.size != 1 or b_scale.size != 1 or y_scale.size != 1) {
        return error.InvalidDimensions;
    }
    if (a_zero_point.size != 1 or b_zero_point.size != 1 or y_zero_point.size != 1) {
        return error.InvalidDimensions;
    }

    const dim_num = a.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        if (a.shape[0] != b.shape[0]) {
            return error.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var output = try Tensor(InputType).fromShape(&allocator, out_shape);
        errdefer output.deinit();

        try qlinear_mat_mul_lean(
            InputType,
            ScaleType,
            ZeroPointType,
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
            &output,
        );

        return output;
    }

    // For tensors with >= 2 dimensions
    if (a.shape[dim_num - 1] != b.shape[dim_num - 2]) {
        return error.InputTensorsWrongShape;
    }

    // Calculate output shape
    const output_shape = try utils.get_qlinearmatmul_output_shape(a.shape, b.shape);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized matrix multiplication
    try qlinear_mat_mul_lean(
        InputType,
        ScaleType,
        ZeroPointType,
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
        &output,
    );

    return output;
}

/// Lean version of QLinearMatMul that operates on pre-allocated output tensor
pub fn qlinear_mat_mul_lean(
    a: anytype,
    a_scale: anytype,
    a_zero_point: anytype,
    b: anytype,
    b_scale: anytype,
    b_zero_point: anytype,
    output: anytype,
    y_scale: anytype,
    y_zero_point: anytype,
) !void {
    const a_scale_val = a_scale.data[0];
    const a_zero_point_val = a_zero_point.data[0];
    const b_scale_val = b_scale.data[0];
    const b_zero_point_val = b_zero_point.data[0];
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = y_zero_point.data[0];

    // Note: Removed pre-calculated scale_factor to match ONNX Runtime's order of operations
    const y_zero_point_f32 = @as(f32, @floatFromInt(@as(i32, y_zero_point_val)));

    const dim_num = a.shape.len;

    // Special case for 1D vectors (dot product)
    if (dim_num == 1) {
        const K = a.shape[0];
        var sum_int: i64 = 0; // Use larger integer for accumulation

        // Perform dot product in integer domain for better precision
        for (0..K) |k| {
            const a_int = @as(i32, a.data[k]) - @as(i32, a_zero_point_val);
            const b_int = @as(i32, b.data[k]) - @as(i32, b_zero_point_val);
            sum_int += @as(i64, a_int) * @as(i64, b_int);
        }

        // Final scaling and quantization - match ONNX Runtime's order of operations
        const sum_float = @as(f32, @floatFromInt(sum_int));
        const scaled_result = (sum_float * a_scale_val * b_scale_val) / y_scale_val + y_zero_point_f32;

        // Clamp to valid range for output type
        const OutputType = @TypeOf(output.data[0]);
        const min_val = if (@typeInfo(OutputType) == .Int)
            @as(f32, @floatFromInt(std.math.minInt(OutputType)))
        else
            std.math.floatMin(OutputType);
        const max_val = if (@typeInfo(OutputType) == .Int)
            @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
        else
            std.math.floatMax(OutputType);
        const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

        output.data[0] = if (@typeInfo(OutputType) == .Int)
            @as(OutputType, @intFromFloat(@round(clamped_result)))
        else
            @as(OutputType, clamped_result);
        return;
    }

    // For matrices and higher-dimensional tensors
    const M = a.shape[dim_num - 2];
    const N = b.shape[dim_num - 1];
    const K = a.shape[dim_num - 1];

    // Calculate batch dimensions
    var batch_size: usize = 1;
    for (0..dim_num - 2) |i| {
        batch_size *= a.shape[i];
    }

    // Calculate strides for each tensor
    const a_batch_stride: usize = M * K;
    const b_batch_stride: usize = K * N;
    const output_batch_stride: usize = M * N;

    // Get output type once for performance
    const OutputType = @TypeOf(output.data[0]);

    // Process each batch
    for (0..batch_size) |batch| {
        const a_batch_offset = batch * a_batch_stride;
        const b_batch_offset = batch * b_batch_stride;
        const output_batch_offset = batch * output_batch_stride;

        // Perform matrix multiplication for this batch
        for (0..M) |i| {
            for (0..N) |j| {
                var sum_int: i64 = 0; // Use larger integer for accumulation

                // Inner product in integer domain
                for (0..K) |k| {
                    const a_idx = a_batch_offset + i * K + k;
                    const b_idx = b_batch_offset + k * N + j;

                    const a_int = @as(i32, a.data[a_idx]) - @as(i32, a_zero_point_val);
                    const b_int = @as(i32, b.data[b_idx]) - @as(i32, b_zero_point_val);
                    sum_int += @as(i64, a_int) * @as(i64, b_int);
                }

                // Match ONNX Runtime's order of operations for better compatibility
                const sum_float = @as(f32, @floatFromInt(sum_int));
                const scaled_result = (sum_float * a_scale_val * b_scale_val) / y_scale_val + y_zero_point_f32;

                // OPTIMIZATION: Direct clamp to output type range for embedded performance
                const output_idx = output_batch_offset + i * N + j;

                if (@typeInfo(OutputType) == .Int) {
                    const min_val = std.math.minInt(OutputType);
                    const max_val = std.math.maxInt(OutputType);
                    const clamped_result = @max(@as(f32, @floatFromInt(min_val)), @min(@as(f32, @floatFromInt(max_val)), @round(scaled_result)));
                    output.data[output_idx] = @as(OutputType, @intFromFloat(clamped_result));
                } else {
                    output.data[output_idx] = @as(OutputType, scaled_result);
                }
            }
        }
    }
}

