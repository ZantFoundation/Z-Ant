const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Returns an allocated slice representing the output shape, or an error.
pub fn get_reshape_output_shape(input_shape: []const usize, new_shape_spec: []const isize, allow_zero: ?bool) ![]usize {
    // Calculate input_size manually
    var input_size: usize = 1;
    for (input_shape) |dim| {
        input_size = std.math.mul(usize, input_size, dim) catch |err| {
            std.log.warn("Error calculating input size (overflow?): {any}\n", .{err});
            return error.Overflow; // Or InvalidDimensions
        };
    }

    // Handle scalar output case
    if (new_shape_spec.len == 0) {
        if (input_size != 1) {
            // Cannot reshape non-scalar to scalar implicitly like this
            return error.InvalidDimensions;
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
                    return error.InvalidDimensions;
                }
                output_shape[i] = input_shape[i];
                // Multiply known_dims_product only if the copied dimension is non-zero
                if (input_shape[i] != 0) {
                    known_dims_product = std.math.mul(usize, known_dims_product, input_shape[i]) catch |err| {
                        std.log.warn("Error calculating known_dims_product (copied dim): {any}\n", .{err});
                        return error.Overflow;
                    };
                } else {
                    // If we copy a zero, the known product becomes zero unless we have a -1
                    known_dims_product = 0;
                }
            }
        } else if (dim_spec == -1) {
            if (neg_one_index != null) {
                // More than one -1 is invalid
                return error.InvalidDimensions;
            }
            neg_one_index = i;
            output_shape[i] = 1; // Placeholder, calculated later
        } else if (dim_spec < 0) {
            // Negative dimensions other than -1 are invalid
            return error.InvalidDimensions;
        } else {
            // Positive dimension
            output_shape[i] = @intCast(dim_spec);
            if (output_shape[i] != 0) {
                known_dims_product = std.math.mul(usize, known_dims_product, output_shape[i]) catch |err| {
                    std.log.warn("Error calculating known_dims_product (positive dim): {any}\n", .{err});
                    return error.Overflow;
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
        return error.InvalidDimensions; // Cannot have explicit 0 and -1 when allowzero=true
    }

    // Second pass: Calculate the inferred dimension if -1 exists
    if (neg_one_index) |idx| {
        if (known_dims_product == 0) {
            // Cannot infer size if product of other dims is 0,
            // unless input_size is also 0.
            if (input_size != 0) {
                return error.InvalidDimensions; // Cannot infer dimension for non-zero input size when other dims product is zero
            } else {
                // If input size is 0 and product is 0, the inferred dim is also 0.
                output_shape[idx] = 0;
            }
        } else {
            if (input_size % known_dims_product != 0) {
                // Input size must be divisible by the product of known dimensions
                return error.InvalidDimensions;
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
            return error.Overflow; // Or InvalidDimensions
        };
    }

    if (input_size != output_size) {
        return error.InvalidDimensions; // Total elements must match
    }

    // Return the successfully calculated shape
    // Note: We allocated output_shape earlier and filled it.
    // The errdefer takes care of freeing if an error occurred *after* allocation.
    // If successful, ownership is transferred to the caller.
    return output_shape;
}

/// Common implementation for reshape_lean functions
pub fn reshape_lean_common(comptime T: anytype, input: *Tensor(T), modified_shape: []usize, neg_one_index: ?usize, known_dims_product: usize, output: *Tensor(T)) !void {
    // If we have a -1 dimension, calculate its size
    if (neg_one_index) |idx| {
        if (known_dims_product == 0) {
            return error.InvalidInput;
        }

        if (input.size % known_dims_product != 0) {
            return error.InputArrayWrongSize;
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
        return error.InputArrayWrongSize;
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
