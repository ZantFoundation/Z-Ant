const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

const reshape_lean_common = @import("utils_reshape.zig").reshape_lean_common;

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// This version accepts a slice of isize for the new shape.
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []const isize, allowZero: ?bool) !Tensor(T) {
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

/// lean version of the reshape function for usize shape arrays
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []const isize, allowZero: ?bool, output: *Tensor(T)) !void {
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
                return error.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim == -1) {
            if (neg_one_index != null) {
                return error.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else if (dim < 0) {
            return error.InvalidInput;
        } else {
            modified_shape[i] = @intCast(dim);
            known_dims_product *= modified_shape[i];
        }
    }

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}
