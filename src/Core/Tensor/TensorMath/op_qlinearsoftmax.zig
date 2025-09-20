const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

/// QLinearSoftmax operation following quantized neural network patterns
/// Performs quantized softmax using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(softmax(dequantize(x)), y_scale, y_zero_point)
pub fn qlinearsoftmax(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    axis: i32,
) !Tensor(InputType) {
    // Input validation
    if (x.size == 0) return TensorError.ZeroSizeTensor;
    if (x.shape.len < 2 or x.shape.len > 5) return TensorError.InvalidDimensions;
    if (x_scale.size != 1 or x_zero_point.size != 1 or
        y_scale.size != 1 or y_zero_point.size != 1)
    {
        return TensorError.InvalidScalarTensor;
    }

    var output = try Tensor(InputType).fromShape(&pkg_allocator, x.shape);
    errdefer output.deinit();

    // Ensure output size matches shape product
    output.size = 1;
    for (x.shape) |d| output.size *= d;

    // Call lean implementation
    try lean_qlinearsoftmax(InputType, ScaleType, ZeroPointType, x, x_scale, x_zero_point, y_scale, y_zero_point, &output, axis);

    return output;
}

/// Lean QLinearSoftmax implementation with pre-allocated output tensor
pub fn lean_qlinearsoftmax(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    output: *Tensor(InputType),
    axis: i32,
) !void {
    const n_dims = x.shape.len;
    // Ensure output size is consistent with input size
    output.size = x.data.len;

    // Normalize axis to positive value
    const normalized_axis: usize = if (axis < 0)
        @intCast(@as(i32, @intCast(n_dims)) + axis)
    else
        @intCast(axis);

    // Validate axis bounds
    if (normalized_axis >= n_dims) return TensorError.InvalidAxis;

    // Extract scalar values
    const x_scale_val = x_scale.data[0];
    const x_zero_point_val = if (@typeInfo(ZeroPointType) == .int)
        @as(i32, @intCast(x_zero_point.data[0]))
    else
        @as(i32, @intFromFloat(x_zero_point.data[0]));
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = if (@typeInfo(ZeroPointType) == .int)
        @as(i32, @intCast(y_zero_point.data[0]))
    else
        @as(i32, @intFromFloat(y_zero_point.data[0]));

    // Convert to float for calculation
    const y_zero_point_f32 = @as(f32, @floatFromInt(y_zero_point_val));

    // Calculate strides for efficient memory access
    var strides: [5]usize = undefined;
    strides[n_dims - 1] = 1;
    var i: usize = n_dims - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * x.shape[i + 1];
    }

    // Calculate dimensions for iteration
    const axis_size = x.shape[normalized_axis];
    const outer_size = calculateOuterSize(x.shape, normalized_axis);
    const inner_size = calculateInnerSize(x.shape, normalized_axis);
    // Guard: ensure output buffer large enough
    if (output.data.len < x.data.len) return TensorError.OutputTensorWrongShape;

    // Debug iteration bounds

    // Allocate temporary arrays for dequantized values
    var dequant_values = try pkg_allocator.alloc(f32, axis_size);
    defer pkg_allocator.free(dequant_values);
    var softmax_values = try pkg_allocator.alloc(f32, axis_size);
    defer pkg_allocator.free(softmax_values);

    // Process each outer×inner combination
    var total_processed: usize = 0;
    for (0..outer_size) |outer_idx| {
        for (0..inner_size) |inner_idx| {
            total_processed += 1;
            // Dequantize values along the axis
            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, x.shape, &strides);
                const InputTypeVal = @TypeOf(x.data[0]);
                const quantized_val = if (@typeInfo(InputTypeVal) == .int)
                    @as(f32, @floatFromInt(x.data[linear_idx]))
                else
                    x.data[linear_idx];

                dequant_values[axis_idx] = (quantized_val - @as(f32, @floatFromInt(x_zero_point_val))) * x_scale_val;
            }

            // DEBUG: check input values variation

            // Debug multiple slices to see input variation
            if ((outer_idx == 0 and inner_idx < 5) or inner_idx == 100 or inner_idx == 143) {}

            // Find maximum value for numerical stability
            var max_val: f32 = -std.math.inf(f32);
            for (dequant_values) |val| {
                if (val > max_val or std.math.isNan(val)) {
                    max_val = val;
                }
            }

            // Compute stabilized exponentials and their sum
            var sum_of_exp: f32 = 0.0;
            for (0..axis_size) |axis_idx| {
                const stabilized_val = dequant_values[axis_idx] - max_val;
                const exp_val = @exp(stabilized_val);
                softmax_values[axis_idx] = exp_val;
                sum_of_exp += exp_val;
            }

            // Normalize and quantize results
            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, x.shape, &strides);

                // Normalize to compute softmax
                const softmax_result = softmax_values[axis_idx] / sum_of_exp;

                // Quantize result
                const scaled_result = (softmax_result / y_scale_val) + y_zero_point_f32;

                // Clamp to valid range for output type
                const OutputType = @TypeOf(output.data[0]);
                const min_val = if (@typeInfo(OutputType) == .int)
                    @as(f32, @floatFromInt(std.math.minInt(OutputType)))
                else
                    std.math.floatMin(OutputType);
                const max_val_out = if (@typeInfo(OutputType) == .int)
                    @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
                else
                    std.math.floatMax(OutputType);
                const clamped_result = std.math.clamp(scaled_result, min_val, max_val_out);

                output.data[linear_idx] = if (@typeInfo(OutputType) == .int)
                    @as(OutputType, @intFromFloat(@round(clamped_result)))
                else
                    @as(OutputType, clamped_result);

                // Debug assignments for multiple ranges
                if (linear_idx < 20 or (linear_idx >= 144 and linear_idx < 154) or (linear_idx >= 280 and linear_idx < 288)) {}
            }

            if (outer_idx == 0 and inner_idx == 0) {
                // DEBUG: output stats for first slice
                var min_o: f32 = std.math.inf(f32);
                var max_o: f32 = -std.math.inf(f32);
                var sum_o: f64 = 0;
                for (0..axis_size) |axis_idx| {
                    const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, x.shape, &strides);
                    const vf: f32 = @as(f32, @floatFromInt(output.data[linear_idx]));
                    if (vf < min_o) min_o = vf;
                    if (vf > max_o) max_o = vf;
                    sum_o += vf;
                }
                _ = @as(f32, @floatCast(sum_o / @as(f64, @floatFromInt(axis_size))));
            }
        }
    }
}

/// Calculate the product of dimensions before the softmax axis
inline fn calculateOuterSize(shape: []const usize, axis: usize) usize {
    var size: usize = 1;
    for (0..axis) |i| {
        size *= shape[i];
    }
    return size;
}

/// Calculate the product of dimensions after the softmax axis
inline fn calculateInnerSize(shape: []const usize, axis: usize) usize {
    var size: usize = 1;
    for ((axis + 1)..shape.len) |i| {
        size *= shape[i];
    }
    return size;
}

/// Calculate linear memory index from multi-dimensional indices using strides
inline fn calculateLinearIndex(outer_idx: usize, axis_idx: usize, inner_idx: usize, n_dims: usize, axis: usize, shape: []const usize, strides: *const [5]usize) usize {
    var linear_idx: usize = 0;
    var remaining_outer = outer_idx;
    var remaining_inner = inner_idx;

    // Handle dimensions before axis (reverse order for proper division)
    if (axis > 0) {
        var dim: usize = axis;
        while (dim > 0) {
            dim -= 1;
            const coord = remaining_outer % shape[dim];
            remaining_outer /= shape[dim];
            linear_idx += coord * strides[dim];
        }
    }

    // Handle the axis dimension
    linear_idx += axis_idx * strides[axis];

    // Handle dimensions after axis (reverse order for proper division)
    if (axis + 1 < n_dims) {
        var dim: usize = n_dims;
        while (dim > axis + 1) {
            dim -= 1;
            const coord = remaining_inner % shape[dim];
            remaining_inner /= shape[dim];
            linear_idx += coord * strides[dim];
        }
    }

    return linear_idx;
}

/// Calculate output shape for QLinearSoftmax - same as input shape
pub fn get_qlinearsoftmax_output_shape(input_shape: []const usize) ![]usize {
    return pkg_allocator.dupe(usize, input_shape);
}
