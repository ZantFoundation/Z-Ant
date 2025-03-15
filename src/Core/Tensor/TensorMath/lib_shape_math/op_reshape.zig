const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// The lean version of this method follows the onnx standard.
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool) !Tensor(T) {
    // Create output tensor with the same size as input but with new shape length
    //std.debug.print("\n[RESHAPE] Input shape: {any}", .{input.shape});
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean_f32(T, input, newShape, allowZero, &output);

    return output;
}

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// This version accepts a slice of usize for the new shape.
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []const usize, allowZero: ?bool) !Tensor(T) {
    // Create output tensor with the same size as input but with new shape length
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean(T, input, newShape, allowZero, &output);

    return output;
}

/// lean version of the reshape function for f32 shape arrays
pub fn reshape_lean_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero;

    // Create a copy of newShape that we can modify
    var modified_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(modified_shape);

    // Track if we have a -1 dimension and its position
    var neg_one_index: ?usize = null;

    // Calculate product of all non-negative and non-zero dimensions
    var known_dims_product: usize = 1;

    // First pass: identify -1 and 0 dimensions
    for (newShape, 0..) |dim, i| {
        if (dim == 0) {
            if (i >= input.shape.len) {
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim < 0) {
            if (neg_one_index != null) {
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else {
            modified_shape[i] = @as(usize, @intFromFloat(dim));
            known_dims_product *= modified_shape[i];
        }
    }

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// lean version of the reshape function for usize shape arrays
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []const usize, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero;

    // Create a copy of newShape that we can modify
    var modified_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(modified_shape);

    // Track if we have a -1 dimension and its position
    var neg_one_index: ?usize = null;

    // Calculate product of all non-negative and non-zero dimensions
    var known_dims_product: usize = 1;

    // First pass: identify -1 and 0 dimensions
    for (newShape, 0..) |dim, i| {
        if (dim == 0) {
            if (i >= input.shape.len) {
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim == std.math.maxInt(usize)) { // Use maxInt as a sentinel for -1
            if (neg_one_index != null) {
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else {
            modified_shape[i] = dim;
            known_dims_product *= modified_shape[i];
        }
    }

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// Common implementation for reshape_lean functions
fn reshape_lean_common(comptime T: anytype, input: *Tensor(T), modified_shape: []usize, neg_one_index: ?usize, known_dims_product: usize, output: *Tensor(T)) !void {
    // If we have a -1 dimension, calculate its size
    if (neg_one_index) |idx| {
        if (known_dims_product == 0) {
            return TensorError.InvalidInput;
        }

        if (input.size % known_dims_product != 0) {
            return TensorError.InputArrayWrongSize;
        }

        modified_shape[idx] = input.size / known_dims_product;
    }

    // Calculate total size of modified shape
    var total_size: usize = 1;
    for (modified_shape) |dim| {
        total_size *= dim;
    }

    // Verify sizes match
    if (total_size != input.size) {
        return TensorError.InputArrayWrongSize;
    }

    // Update output shape
    for (modified_shape, 0..) |dim, i| {
        output.shape[i] = dim;
    }
    output.size = total_size;

    // Copy data from input to output
    @memcpy(output.data, input.data);
}
