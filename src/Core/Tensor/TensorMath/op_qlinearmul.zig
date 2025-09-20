const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing multiplication operation for broadcasting logic
const multiplication = @import("lib_elementWise_math/op_multiplication.zig");

/// QLinearMul operation following ONNX specification
/// Performs quantized element-wise multiplication using linear quantization scheme
///
/// INPUTS:
/// - a: quantized input tensor A (typically int8/uint8)
/// - a_scale: scale factor for input A quantization
/// - a_zero_point: zero point for input A quantization
/// - b: quantized input tensor B
/// - b_scale: scale factor for input B quantization
/// - b_zero_point: zero point for input B quantization
/// - c_scale: scale factor for output quantization
/// - c_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - c: quantized output tensor
///
/// Formula: quantized_output = quantize(dequantize(a) * dequantize(b), c_scale, c_zero_point)
pub fn qlinearmul(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    a: *const Tensor(InputType),
    a_scale: *const Tensor(ScaleType),
    a_zero_point: *const Tensor(ZeroPointType),
    b: *const Tensor(InputType),
    b_scale: *const Tensor(ScaleType),
    b_zero_point: *const Tensor(ZeroPointType),
    c_scale: *const Tensor(ScaleType),
    c_zero_point: *const Tensor(ZeroPointType),
) !Tensor(InputType) {
    // Input validation
    if (a.size == 0 or b.size == 0) return TensorError.ZeroSizeTensor;
    if (a_scale.size != 1 or a_zero_point.size != 1 or
        b_scale.size != 1 or b_zero_point.size != 1 or
        c_scale.size != 1 or c_zero_point.size != 1)
    {
        return TensorError.InvalidScalarTensor;
    }

    // Calculate output shape using broadcasting
    const output_shape = try multiplication.get_mul_output_shape(a.shape, b.shape);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Call lean implementation
    try lean_qlinearmul(InputType, ScaleType, ZeroPointType, a, a_scale, a_zero_point, b, b_scale, b_zero_point, c_scale, c_zero_point, &output);

    return output;
}

/// Lean QLinearMul implementation with pre-allocated output tensor
pub fn lean_qlinearmul(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    a: *const Tensor(InputType),
    a_scale: *const Tensor(ScaleType),
    a_zero_point: *const Tensor(ZeroPointType),
    b: *const Tensor(InputType),
    b_scale: *const Tensor(ScaleType),
    b_zero_point: *const Tensor(ZeroPointType),
    c_scale: *const Tensor(ScaleType),
    c_zero_point: *const Tensor(ZeroPointType),
    output: *Tensor(InputType),
) !void {
    // Extract scalar values
    const a_scale_val = a_scale.data[0];
    const a_zero_point_val = if (@typeInfo(ZeroPointType) == .int)
        @as(i32, @intCast(a_zero_point.data[0]))
    else
        @as(i32, @intFromFloat(a_zero_point.data[0]));
    const b_scale_val = b_scale.data[0];
    const b_zero_point_val = if (@typeInfo(ZeroPointType) == .int)
        @as(i32, @intCast(b_zero_point.data[0]))
    else
        @as(i32, @intFromFloat(b_zero_point.data[0]));
    const c_scale_val = c_scale.data[0];
    const c_zero_point_val = if (@typeInfo(ZeroPointType) == .int)
        @as(i32, @intCast(c_zero_point.data[0]))
    else
        @as(i32, @intFromFloat(c_zero_point.data[0]));

    // Convert to float for calculation
    const c_zero_point_f32 = @as(f32, @floatFromInt(c_zero_point_val));

    // Simple case: same shape tensors
    if (std.mem.eql(usize, a.shape, b.shape) and a.size == output.size) {
        for (0..@min(a.size, @min(b.size, output.size))) |i| {
            // Dequantize inputs
            const AType = @TypeOf(a.data[0]);
            const BType = @TypeOf(b.data[0]);
            const OutputType = @TypeOf(output.data[0]);

            const a_dequant = if (@typeInfo(AType) == .int)
                (@as(f32, @floatFromInt(a.data[i])) - @as(f32, @floatFromInt(a_zero_point_val))) * a_scale_val
            else
                (a.data[i] - @as(AType, @floatFromInt(a_zero_point_val))) * a_scale_val;
            const b_dequant = if (@typeInfo(BType) == .int)
                (@as(f32, @floatFromInt(b.data[i])) - @as(f32, @floatFromInt(b_zero_point_val))) * b_scale_val
            else
                (b.data[i] - @as(BType, @floatFromInt(b_zero_point_val))) * b_scale_val;

            // Multiply in float domain
            const result_float = a_dequant * b_dequant;

            // Quantize result
            const scaled_result = (result_float / c_scale_val) + c_zero_point_f32;

            // Clamp to valid range for output type
            const min_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.minInt(OutputType)))
            else
                std.math.floatMin(OutputType);
            const max_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
            else
                std.math.floatMax(OutputType);
            const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

            output.data[i] = if (@typeInfo(OutputType) == .int)
                @as(OutputType, @intFromFloat(@round(clamped_result)))
            else
                @as(OutputType, clamped_result);
        }
    } else {
        // Broadcasting case - calculate strides for each input tensor
        const max_rank = @max(@max(a.shape.len, b.shape.len), output.shape.len);

        // Allocate arrays for broadcasting calculations
        var loop_indices = try pkg_allocator.alloc(usize, max_rank);
        defer pkg_allocator.free(loop_indices);
        var strides1 = try pkg_allocator.alloc(usize, max_rank);
        defer pkg_allocator.free(strides1);
        var strides2 = try pkg_allocator.alloc(usize, max_rank);
        defer pkg_allocator.free(strides2);
        var out_strides = try pkg_allocator.alloc(usize, max_rank);
        defer pkg_allocator.free(out_strides);

        // Initialize arrays
        @memset(loop_indices, 0);
        @memset(strides1, 0);
        @memset(strides2, 0);
        @memset(out_strides, 0);

        // Calculate strides for each tensor using broadcasting rules
        var stride1: usize = 1;
        var stride2: usize = 1;
        var out_stride: usize = 1;

        var dim_idx: usize = max_rank;
        while (dim_idx > 0) {
            dim_idx -= 1;

            // Get dimension sizes (1 if dimension doesn't exist)
            const a_dim_idx = if (dim_idx + a.shape.len >= max_rank)
                dim_idx + a.shape.len - max_rank
            else
                null;
            const b_dim_idx = if (dim_idx + b.shape.len >= max_rank)
                dim_idx + b.shape.len - max_rank
            else
                null;
            const out_dim_idx = if (dim_idx + output.shape.len >= max_rank)
                dim_idx + output.shape.len - max_rank
            else
                null;

            const a_dim_size = if (a_dim_idx) |idx| a.shape[idx] else 1;
            const b_dim_size = if (b_dim_idx) |idx| b.shape[idx] else 1;
            const output_dim_size = if (out_dim_idx) |idx| output.shape[idx] else 1;

            // Set strides (0 for broadcasted dimensions)
            strides1[dim_idx] = if (a_dim_size == 1 and output_dim_size > 1) 0 else stride1;
            strides2[dim_idx] = if (b_dim_size == 1 and output_dim_size > 1) 0 else stride2;
            out_strides[dim_idx] = out_stride;

            // Update strides for next iteration
            stride1 *= a_dim_size;
            stride2 *= b_dim_size;
            out_stride *= output_dim_size;
        }

        // Perform broadcasted multiplication
        for (0..output.size) |i| {
            // Calculate multi-dimensional indices for current output position
            var current_flat_index = i;
            for (0..max_rank) |dim| {
                const current_stride = out_strides[dim];
                loop_indices[dim] = @divFloor(current_flat_index, current_stride);
                current_flat_index = @mod(current_flat_index, current_stride);
            }

            // Calculate linear input indices using broadcasting
            var idx1: usize = 0;
            var idx2: usize = 0;
            for (0..max_rank) |dim| {
                const current_loop_index = loop_indices[dim];
                idx1 += current_loop_index * strides1[dim];
                idx2 += current_loop_index * strides2[dim];
            }

            // Bounds checking to prevent out of bounds access
            if (idx1 >= a.size or idx2 >= b.size) {
                return TensorMathError.IndexOutOfBounds;
            }

            // Dequantize inputs
            const AType = @TypeOf(a.data[0]);
            const BType = @TypeOf(b.data[0]);
            const OutputType = @TypeOf(output.data[0]);

            const a_dequant = if (@typeInfo(AType) == .int)
                (@as(f32, @floatFromInt(a.data[idx1])) - @as(f32, @floatFromInt(a_zero_point_val))) * a_scale_val
            else
                (a.data[idx1] - @as(AType, @floatFromInt(a_zero_point_val))) * a_scale_val;
            const b_dequant = if (@typeInfo(BType) == .int)
                (@as(f32, @floatFromInt(b.data[idx2])) - @as(f32, @floatFromInt(b_zero_point_val))) * b_scale_val
            else
                (b.data[idx2] - @as(BType, @floatFromInt(b_zero_point_val))) * b_scale_val;

            // Multiply in float domain
            const result_float = a_dequant * b_dequant;

            // Quantize result
            const scaled_result = (result_float / c_scale_val) + c_zero_point_f32;

            // Clamp to valid range for output type
            const min_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.minInt(OutputType)))
            else
                std.math.floatMin(OutputType);
            const max_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
            else
                std.math.floatMax(OutputType);
            const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

            output.data[i] = if (@typeInfo(OutputType) == .int)
                @as(OutputType, @intFromFloat(@round(clamped_result)))
            else
                @as(OutputType, clamped_result);
        }
    }

    // Output stats
    var min_o: f32 = std.math.inf(f32);
    var max_o: f32 = -std.math.inf(f32);
    var sum_o: f64 = 0;
    for (output.data) |v| {
        const vf: f32 = @as(f32, @floatFromInt(v));
        if (vf < min_o) min_o = vf;
        if (vf > max_o) max_o = vf;
        sum_o += vf;
    }
    _ = @as(f32, @floatCast(sum_o / @as(f64, @floatFromInt(output.data.len))));
}

/// Calculate output shape for QLinearMul - same as regular Mul (uses broadcasting)
pub fn get_qlinearmul_output_shape(
    a_shape: []const usize,
    b_shape: []const usize,
) ![]usize {
    return multiplication.get_mul_output_shape(a_shape, b_shape);
}
