const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

/// Calculate output shape for QLinearMatMul - same as regular MatMul
pub fn get_qlinearmatmul_output_shape(
    a_shape: []const usize,
    b_shape: []const usize,
) ![]usize {
    if (a_shape.len != b_shape.len) {
        return error.InputTensorDifferentShape;
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
            return error.InputTensorsWrongShape;
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
