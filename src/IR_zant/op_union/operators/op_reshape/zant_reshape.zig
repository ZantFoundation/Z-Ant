const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const UOpBuilder = zant.uops.UOpBuilder;
const DType = zant.uops.DType;
const Any = zant.uops.Any;

const pkg_allocator = zant.utils.allocator.allocator;

// Temprorary import for testing
const lowerNeg = @import("../op_neg/zant_neg.zig").lowerNeg;

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// The lean version of this method follows the onnx standard.
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool) !Tensor(T) {
    //std.log.debug("\n[RESHAPE_F32] Input shape: {any}, newShape: {any}\n", .{ input.shape, newShape });

    // Create output tensor with the same size as input but with new shape length
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    //std.log.debug("[RESHAPE_F32] Temp shape: {any}\n", .{temp_shape});

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean_f32(T, input, newShape, allowZero, &output);

    //std.log.debug("[RESHAPE_F32] Output shape: {any}, size: {}\n", .{ output.shape, output.size });
    return output;
}

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// This version accepts a slice of isize for the new shape.
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []const isize, allowZero: ?bool) !Tensor(T) {
    //std.log.debug("\n[RESHAPE] Input shape: {any}, newShape: {any}\n", .{ input.shape, newShape });

    // Create output tensor with the same size as input but with new shape length
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    //std.log.debug("[RESHAPE] Temp shape: {any}\n", .{temp_shape});

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean(T, input, newShape, allowZero, &output);

    //std.log.debug("[RESHAPE] Output shape: {any}, size: {}\n", .{ output.shape, output.size });
    return output;
}

/// lean version of the reshape function for f32 shape arrays
pub fn reshape_lean_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool, output: *Tensor(T)) !void {
    //std.log.debug("\n[RESHAPE_LEAN_F32] Input shape: {any}, newShape: {any}\n", .{ input.shape, newShape });
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
                //std.log.debug("[RESHAPE_LEAN_F32] Error: Invalid input - dim is 0 but index {} >= input shape len {}\n", .{ i, input.shape.len });
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim < 0) {
            if (neg_one_index != null) {
                //std.log.debug("[RESHAPE_LEAN_F32] Error: Invalid input - multiple negative dimensions\n", .{});
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else {
            modified_shape[i] = @as(usize, @intFromFloat(dim));
            known_dims_product *= modified_shape[i];
        }
    }

    //std.log.debug("[RESHAPE_LEAN_F32] Modified shape before inference: {any}, neg_one_index: {?}, known_dims_product: {}\n", .{ modified_shape, neg_one_index, known_dims_product });

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// lean version of the reshape function for usize shape arrays
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []const isize, allowZero: ?bool, output: *Tensor(T)) !void {
    //std.log.debug("\n[RESHAPE_LEAN] Input shape: {any}, newShape: {any}\n", .{ input.shape, newShape });
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
                //std.log.debug("[RESHAPE_LEAN] Error: Invalid input - dim is 0 but index {} >= input shape len {}\n", .{ i, input.shape.len });
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim == -1) {
            if (neg_one_index != null) {
                //std.log.debug("[RESHAPE_LEAN] Error: Invalid input - multiple negative dimensions\n", .{});
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else if (dim < 0) {
            //std.log.debug("[RESHAPE_LEAN] Error: Invalid input - negative dimension other than -1\n", .{});
            return TensorError.InvalidInput;
        } else {
            modified_shape[i] = @intCast(dim);
            known_dims_product *= modified_shape[i];
        }
    }

    //std.log.debug("[RESHAPE_LEAN] Modified shape before inference: {any}, neg_one_index: {?}, known_dims_product: {}\n", .{ modified_shape, neg_one_index, known_dims_product });

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// Returns an allocated slice representing the output shape, or an error.
pub fn get_reshape_output_shape(input_shape: []const usize, new_shape_spec: []const isize, allow_zero: ?bool) ![]usize {
    // Calculate input_size manually
    var input_size: usize = 1;
    for (input_shape) |dim| {
        input_size = std.math.mul(usize, input_size, dim) catch |err| {
            std.log.warn("Error calculating input size (overflow?): {any}\n", .{err});
            return TensorMathError.Overflow; // Or InvalidDimensions
        };
    }

    // Handle scalar output case
    if (new_shape_spec.len == 0) {
        if (input_size != 1) {
            // Cannot reshape non-scalar to scalar implicitly like this
            return TensorMathError.InvalidDimensions;
        }
        // Return an empty slice for scalar shape
        return pkg_allocator.alloc(usize, 0);
    }

    var output_shape = try pkg_allocator.alloc(usize, new_shape_spec.len);
    errdefer pkg_allocator.free(output_shape); // Ensure cleanup on error during calculation

    var neg_one_index: ?usize = null;
    var known_dims_product: usize = 1;
    var has_explicit_zero: bool = false;

    // First pass: Process dimensions, identify -1, handle 0 based on allow_zero
    for (new_shape_spec, 0..) |dim_spec, i| {
        if (dim_spec == 0) {
            if (allow_zero orelse false) {
                // allowzero is true, dimension is explicitly 0
                output_shape[i] = 0;
                has_explicit_zero = true;
                // known_dims_product remains unchanged (effectively multiplied by 0 later)
            } else {
                // allowzero is false/null, copy dimension from input
                if (i >= input_shape.len) {
                    // Cannot copy dimension if index is out of bounds
                    return TensorMathError.InvalidDimensions;
                }
                output_shape[i] = input_shape[i];
                // Multiply known_dims_product only if the copied dimension is non-zero
                if (input_shape[i] != 0) {
                    known_dims_product = std.math.mul(usize, known_dims_product, input_shape[i]) catch |err| {
                        std.log.warn("Error calculating known_dims_product (copied dim): {any}\n", .{err});
                        return TensorMathError.Overflow;
                    };
                } else {
                    // If we copy a zero, the known product becomes zero unless we have a -1
                    known_dims_product = 0;
                }
            }
        } else if (dim_spec == -1) {
            if (neg_one_index != null) {
                // More than one -1 is invalid
                return TensorMathError.InvalidDimensions;
            }
            neg_one_index = i;
            output_shape[i] = 1; // Placeholder, calculated later
        } else if (dim_spec < 0) {
            // Negative dimensions other than -1 are invalid
            return TensorMathError.InvalidDimensions;
        } else {
            // Positive dimension
            output_shape[i] = @intCast(dim_spec);
            if (output_shape[i] != 0) {
                known_dims_product = std.math.mul(usize, known_dims_product, output_shape[i]) catch |err| {
                    std.log.warn("Error calculating known_dims_product (positive dim): {any}\n", .{err});
                    return TensorMathError.Overflow;
                };
            } else {
                // If we have an explicit zero (dim_spec > 0 but cast to 0?), treat as explicit zero
                has_explicit_zero = true;
                known_dims_product = 0;
            }
        }
    }

    // Check for conflict: allowzero=true and both 0 and -1 present
    if ((allow_zero orelse false) and has_explicit_zero and neg_one_index != null) {
        return TensorMathError.InvalidDimensions; // Cannot have explicit 0 and -1 when allowzero=true
    }

    // Second pass: Calculate the inferred dimension if -1 exists
    if (neg_one_index) |idx| {
        if (known_dims_product == 0) {
            // Cannot infer size if product of other dims is 0,
            // unless input_size is also 0.
            if (input_size != 0) {
                return TensorMathError.InvalidDimensions; // Cannot infer dimension for non-zero input size when other dims product is zero
            } else {
                // If input size is 0 and product is 0, the inferred dim is also 0.
                output_shape[idx] = 0;
            }
        } else {
            if (input_size % known_dims_product != 0) {
                // Input size must be divisible by the product of known dimensions
                return TensorMathError.InvalidDimensions;
            }
            output_shape[idx] = input_size / known_dims_product;
        }
    }

    // Final check: Verify the total size of the calculated output shape matches the input size
    // Calculate output_size manually
    var output_size: usize = 1;
    for (output_shape) |dim| {
        output_size = std.math.mul(usize, output_size, dim) catch |err| {
            std.log.warn("Error calculating output size (overflow?): {any}\n", .{err});
            // Don't free output_shape here, the errdefer above will handle it.
            return TensorMathError.Overflow; // Or InvalidDimensions
        };
    }

    if (input_size != output_size) {
        return TensorMathError.InvalidDimensions; // Total elements must match
    }

    // Return the successfully calculated shape
    // Note: We allocated output_shape earlier and filled it.
    // The errdefer takes care of freeing if an error occurred *after* allocation.
    // If successful, ownership is transferred to the caller.
    return output_shape;
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

    // Handle the shape - manage memory correctly
    if (output.shape.len != modified_shape.len) {
        // If lengths differ, free the old shape and allocate a new one
        pkg_allocator.free(output.shape);
        output.shape = try pkg_allocator.dupe(usize, modified_shape);
    } else {
        // If lengths match, just copy the new values
        for (modified_shape, 0..) |dim, i| {
            output.shape[i] = dim;
        }
    }

    // Ensure output.size matches the size calculated from the shape
    output.size = total_size;

    // Copy input data to output - manage memory correctly
    if (output.data.len != input.data.len) {
        // If lengths differ, free the old data and allocate new memory
        pkg_allocator.free(output.data);
        output.data = try pkg_allocator.dupe(T, input.data);
    } else {
        // If lengths match, copy the data
        @memcpy(output.data, input.data);
    }
}
