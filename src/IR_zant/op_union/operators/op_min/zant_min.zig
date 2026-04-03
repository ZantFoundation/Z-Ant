const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- MIN OPERATOR ---------------------

/// Computes the output shape for the Min operator.
/// Min is an element-wise operation, so output shape is the broadcast shape of all inputs.
pub fn get_min_output_shape(input_shapes: []const []const usize) ![]usize {
    if (input_shapes.len == 0) return TensorMathError.InvalidDimensions;
    if (input_shapes.len == 1) {
        const output_shape = try pkg_allocator.alloc(usize, input_shapes[0].len);
        @memcpy(output_shape, input_shapes[0]);
        return output_shape;
    }

    // For multiple inputs, we need to compute broadcast shape
    // For simplicity, assume all inputs have the same shape (most common case)
    // TODO: Implement proper broadcasting logic
    const output_shape = try pkg_allocator.alloc(usize, input_shapes[0].len);
    @memcpy(output_shape, input_shapes[0]);
    return output_shape;
}

/// Applies the min function element-wise across multiple tensors, allocating a new output tensor.
pub fn min(comptime T: anytype, inputs: []*Tensor(T)) !Tensor(T) {
    if (inputs.len == 0) return TensorMathError.InvalidDimensions;
    if (inputs.len == 1) return inputs[0].copy();

    // For now, assume all tensors have the same shape
    const output_shape = try get_min_output_shape(&[_][]const usize{inputs[0].shape});
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(inputs[0].allocator, output_shape);
    errdefer output.deinit();

    try min_lean(T, inputs, &output);
    return output;
}

/// Lean version that computes min in-place using a pre-allocated output tensor.
pub fn min_lean(comptime T: anytype, inputs: []*Tensor(T), output: *Tensor(T)) !void {
    if (inputs.len == 0) return TensorMathError.InvalidDimensions;
    if (inputs.len == 1) {
        @memcpy(output.data, inputs[0].data);
        return;
    }

    // Start with the first tensor
    @memcpy(output.data, inputs[0].data);

    // Apply min with each subsequent tensor
    for (inputs[1..]) |input| {
        if (input.size != output.size) return TensorMathError.InvalidDimensions;

        for (0..output.size) |i| {
            if (input.data[i] < output.data[i]) {
                output.data[i] = input.data[i];
            }
        }
    }
}

/// Computes element-wise minimum between two tensors, allocating a new output tensor.
pub fn min_two(comptime T: anytype, a: *Tensor(T), b: *Tensor(T)) !Tensor(T) {
    if (a.size != b.size) return TensorMathError.InvalidDimensions;

    var output = try Tensor(T).fromShape(a.allocator, a.shape);
    errdefer output.deinit();

    try min_two_lean(T, a, b, &output);
    return output;
}

/// Lean version for two tensors that computes min in-place using a pre-allocated output tensor.
pub fn min_two_lean(comptime T: anytype, a: *Tensor(T), b: *Tensor(T), output: *Tensor(T)) !void {
    if (a.size != b.size or a.size != output.size) return TensorMathError.InvalidDimensions;

    for (0..output.size) |i| {
        output.data[i] = if (a.data[i] < b.data[i]) a.data[i] else b.data[i];
    }
}

/// Reduces a tensor along specified axes to find minimum values.
pub fn reduce_min(comptime T: anytype, input: *Tensor(T), axes: ?[]const i64, keepdims: bool) !Tensor(T) {
    if (axes == null) {
        // Reduce over all dimensions - find global minimum
        var output_shape: []usize = undefined;
        if (keepdims) {
            output_shape = try pkg_allocator.alloc(usize, input.shape.len);
            @memset(output_shape, 1);
        } else {
            output_shape = try pkg_allocator.alloc(usize, 0);
        }

        var output = try Tensor(T).fromShape(input.allocator, output_shape);
        pkg_allocator.free(output_shape);
        errdefer output.deinit();

        if (input.size == 0) return TensorMathError.InvalidDimensions;

        var min_val = input.data[0];
        for (input.data[1..]) |val| {
            if (val < min_val) min_val = val;
        }

        if (output.size > 0) output.data[0] = min_val;
        return output;
    }

    // For specific axes, we'd need more complex reduction logic
    // For now, return a copy (placeholder implementation)
    return input.copy();
}

/// Lean version of reduce_min that operates on pre-allocated output tensor.
pub fn reduce_min_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T), axes: ?[]const i64, _: bool) !void {
    if (axes == null) {
        // Global reduction
        if (input.size == 0) return TensorMathError.InvalidDimensions;

        var min_val = input.data[0];
        for (input.data[1..]) |val| {
            if (val < min_val) min_val = val;
        }

        if (output.size > 0) output.data[0] = min_val;
        return;
    }

    // For specific axes, placeholder implementation
    if (input.size == output.size) {
        @memcpy(output.data, input.data);
    }
}
