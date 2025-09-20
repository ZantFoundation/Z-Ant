const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing matmul operation for shape calculation
const matmul = @import("op_mat_mul.zig");

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
pub fn qlinearmatmul(
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
        return TensorMathError.InputTensorDifferentShape;
    }
    if (a_scale.size != 1 or b_scale.size != 1 or y_scale.size != 1) {
        return TensorMathError.InvalidDimensions;
    }
    if (a_zero_point.size != 1 or b_zero_point.size != 1 or y_zero_point.size != 1) {
        return TensorMathError.InvalidDimensions;
    }

    const dim_num = a.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        if (a.shape[0] != b.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var output = try Tensor(InputType).fromShape(&allocator, out_shape);
        errdefer output.deinit();

        try lean_qlinearmatmul(
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
        return TensorMathError.InputTensorsWrongShape;
    }

    // Calculate output shape
    const output_shape = try get_qlinearmatmul_output_shape(a.shape, b.shape);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized matrix multiplication
    try lean_qlinearmatmul(
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
pub fn lean_qlinearmatmul(
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

/// Calculate output shape for QLinearMatMul - same as regular MatMul
pub fn get_qlinearmatmul_output_shape(
    a_shape: []const usize,
    b_shape: []const usize,
) ![]usize {
    if (a_shape.len != b_shape.len) {
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = a_shape.len;

    // Special case for 1D vectors
    if (dim_num == 1) {
        const output_shape = try pkg_allocator.alloc(usize, 1);
        output_shape[0] = 1;
        return output_shape;
    }

    // For higher-dimensional tensors
    const output_shape = try pkg_allocator.alloc(usize, dim_num);

    // Copy batch dimensions
    for (0..dim_num - 2) |i| {
        if (a_shape[i] != b_shape[i]) {
            pkg_allocator.free(output_shape);
            return TensorMathError.InputTensorsWrongShape;
        }
        output_shape[i] = a_shape[i];
    }

    // Set matrix dimensions
    output_shape[dim_num - 2] = a_shape[dim_num - 2]; // M
    output_shape[dim_num - 1] = b_shape[dim_num - 1]; // N

    return output_shape;
}

/// QGemm variant of QLinearMatMul that handles transposed weights - OPTIMIZED FOR EMBEDDED
/// For QGemm, B tensor is typically [N, K] instead of [K, N], so we transpose implicitly
pub fn qgemm_lean(
    a: anytype,
    a_scale: anytype,
    a_zero_point: anytype,
    b: anytype, // Weights tensor, typically [N, K] format
    b_scale: anytype,
    b_zero_point: anytype,
    output: anytype,
    y_scale: anytype,
    y_zero_point: anytype,
) !void {
    const OutputType = @TypeOf(output.data[0]);

    // Get scalar values from scale and zero_point tensors
    const a_scale_val = a_scale.data[0];
    const a_zero_point_val = @as(i32, a_zero_point.data[0]);
    const b_scale_val = b_scale.data[0];
    const b_zero_point_val = @as(i32, b_zero_point.data[0]);
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = @as(i32, y_zero_point.data[0]);

    // Note: Removed pre-calculated scale_factor to match ONNX Runtime's order of operations
    const y_zero_point_f32 = @as(f32, @floatFromInt(y_zero_point_val));

    const dim_num = a.shape.len;

    // For QGemm: A is [M, K], B is [N, K] (transposed), Output is [M, N]
    const M = a.shape[dim_num - 2];
    const K = a.shape[dim_num - 1];
    const N = b.shape[dim_num - 2]; // Note: N comes from first dimension of B

    // Calculate batch dimensions
    var batch_size: usize = 1;
    for (0..dim_num - 2) |i| {
        batch_size *= a.shape[i];
    }

    // Calculate strides
    const a_batch_stride: usize = M * K;
    const b_batch_stride: usize = N * K; // B is [N, K] format
    const output_batch_stride: usize = M * N;

    // Process each batch
    for (0..batch_size) |batch| {
        const a_batch_offset = batch * a_batch_stride;
        const b_batch_offset = batch * b_batch_stride;
        const output_batch_offset = batch * output_batch_stride;

        // Perform matrix multiplication: A * B^T
        for (0..M) |i| {
            for (0..N) |j| {
                var sum_int: i64 = 0;

                // OPTIMIZATION: Inner product stays in integer domain
                for (0..K) |k| {
                    const a_idx = a_batch_offset + i * K + k;
                    const b_idx = b_batch_offset + j * K + k; // Transposed access: B[j][k]

                    const a_int = @as(i32, a.data[a_idx]) - a_zero_point_val;
                    const b_int = @as(i32, b.data[b_idx]) - b_zero_point_val;
                    sum_int += @as(i64, a_int) * @as(i64, b_int);
                }

                // Match ONNX Runtime's order of operations for better compatibility
                const sum_float = @as(f32, @floatFromInt(sum_int));
                const scaled_result = (sum_float * a_scale_val * b_scale_val) / y_scale_val + y_zero_point_f32;

                // OPTIMIZATION: Direct clamp to output type range for embedded
                const output_idx = output_batch_offset + i * N + j;

                if (@typeInfo(OutputType) == .int) {
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
