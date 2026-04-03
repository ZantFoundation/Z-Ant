const std = @import("std");
const zant = @import("../../../../zant.zig");
const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

/// The Softmax activation function is used in multi-class classification tasks to convert
/// logits (raw output values) into probabilities that sum to 1.
/// Ideal for output layers in multi-class neural networks.
///
/// Implements ONNX Softmax specification:
/// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
/// Default axis is -1 (last dimension)
pub fn softmax(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    return softmax_with_axis(T, tensor, -1);
}

/// Softmax with configurable axis parameter following ONNX specification
pub fn softmax_with_axis(comptime T: anytype, tensor: *Tensor(T), axis: i32) !Tensor(T) {

    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;
    if (tensor.shape.len < 2 or tensor.shape.len > 5) return TensorError.InvalidDimensions;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    //compute
    try lean_softmax_with_axis(T, tensor, &output_tensor, axis);
    return output_tensor;
}

pub inline fn lean_softmax(comptime T: anytype, input: *Tensor(T), output: *Tensor(T), axis: i32) !void {
    try lean_softmax_with_axis(T, input, output, axis);
}

/// Optimized softmax implementation following ONNX specification with stride-aware computation
/// Supports tensors of dimensions 2-5 and any axis parameter
pub inline fn lean_softmax_with_axis(comptime T: anytype, input: *Tensor(T), output: *Tensor(T), axis: i32) !void {
    const n_dims = input.shape.len;

    // Normalize axis to positive value
    const normalized_axis: usize = if (axis < 0)
        @intCast(@as(i32, @intCast(n_dims)) + axis)
    else
        @intCast(axis);

    // Validate axis bounds
    if (normalized_axis > n_dims) return TensorError.InvalidAxis;

    // Calculate strides for efficient memory access
    var strides: [5]usize = undefined;
    strides[n_dims - 1] = 1;
    var i: usize = n_dims - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    // Calculate dimensions for iteration
    const axis_size = input.shape[normalized_axis];
    const outer_size = calculateOuterSize(input.shape, normalized_axis);
    const inner_size = calculateInnerSize(input.shape, normalized_axis);

    // Process each outer×inner combination
    for (0..outer_size) |outer_idx| {
        for (0..inner_size) |inner_idx| {
            // Find maximum value along the axis for numerical stability
            var max_val: T = -std.math.inf(T);
            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, input.shape, &strides);
                const val = input.data[linear_idx];
                if (val > max_val or std.math.isNan(val)) {
                    max_val = val;
                }
            }

            // Compute stabilized exponentials and their sum
            var sum_of_exp: T = 0.0;
            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, input.shape, &strides);
                const stabilized_val = input.data[linear_idx] - max_val;
                const exp_val = @exp(stabilized_val);
                output.data[linear_idx] = exp_val;
                sum_of_exp += exp_val;
            }

            // Normalize to compute softmax
            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, n_dims, normalized_axis, input.shape, &strides);
                output.data[linear_idx] /= sum_of_exp;
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

    // Handle dimensions before axis
    var dim: usize = 0;
    while (dim < axis) : (dim += 1) {
        const coord = remaining_outer % shape[dim];
        remaining_outer /= shape[dim];
        linear_idx += coord * strides[dim];
    }

    // Handle the axis dimension
    linear_idx += axis_idx * strides[axis];

    // Handle dimensions after axis
    dim = axis + 1;
    while (dim < n_dims) : (dim += 1) {
        const coord = remaining_inner % shape[dim];
        remaining_inner /= shape[dim];
        linear_idx += coord * strides[dim];
    }

    return linear_idx;
}
