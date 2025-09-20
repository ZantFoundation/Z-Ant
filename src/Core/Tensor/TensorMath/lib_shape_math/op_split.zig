const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Split a tensor into multiple tensors along a specified axis.
/// If split_sizes is null, the tensor is split into equal parts.
/// If split_sizes is provided, it specifies the size of each split.
/// Negative axis values count from the back (-1 means last axis).
/// Returns an array of tensors that must be freed by the caller.
pub fn split(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const usize) ![]Tensor(T) {
    // Handle negative axis
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    if (positive_axis >= t.shape.len) return TensorError.InvalidAxis;

    // Calculate split sizes
    const dim_size = t.shape[positive_axis];
    var sizes = std.ArrayList(usize).init(t.allocator.*);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
    }

    // Create output tensors
    var output_tensors = try t.allocator.alloc(Tensor(T), sizes.items.len);
    errdefer {
        for (output_tensors) |*tensor| {
            tensor.deinit();
        }
        t.allocator.free(output_tensors);
    }

    // Create a durable copy of the split sizes
    const durable_split_sizes = try t.allocator.dupe(usize, sizes.items);
    defer t.allocator.free(durable_split_sizes);

    try split_lean(T, t, axis, durable_split_sizes, &output_tensors);

    return output_tensors;
}

//lean split
//inputs:
//split_sizes can't be null
pub fn split_lean(comptime T: type, input_tensor: *Tensor(T), axis: i64, split_sizes: []const usize, output_tensors: *[]Tensor(T)) !void {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_tensor.shape.len)) + axis;
        if (adjusted < 0) return TensorError.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_tensor.shape.len) return TensorError.InvalidAxis;

    // Get split output shapes
    const output_shapes = try get_split_output_shapes(input_tensor.shape, axis, split_sizes, output_tensors.len);
    defer {
        // Free the individual shape arrays and the output_shapes array
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Ensure we have enough output tensors
    if (output_tensors.len != output_shapes.len) {
        return TensorError.InvalidInput;
    }

    // Initialize output tensors with proper shapes
    for (output_shapes, 0..) |shape, i| {
        // Calculate required size
        var total_size: usize = 1;
        for (shape) |dim| {
            total_size *= dim;
        }

        // Initialize the tensor properly
        output_tensors.*[i] = Tensor(T){
            .data = try input_tensor.allocator.alloc(T, total_size),
            .shape = try input_tensor.allocator.dupe(usize, shape),
            .size = total_size,
            .allocator = input_tensor.allocator,
        };
    }

    // Copy data from input tensor to output tensors
    const offsets = try compute_split_offsets(input_tensor.shape, positive_axis, split_sizes, output_tensors.len);
    defer input_tensor.allocator.free(offsets);

    // Now let's implement the actual data copying
    // Calculate the size of each dimension
    var dim_sizes = try input_tensor.allocator.alloc(usize, input_tensor.shape.len);
    defer input_tensor.allocator.free(dim_sizes);

    // Calculate size of each dimension (for faster indexing)
    dim_sizes[input_tensor.shape.len - 1] = 1;
    var i: usize = input_tensor.shape.len - 1;
    while (i > 0) {
        i -= 1;
        dim_sizes[i] = dim_sizes[i + 1] * input_tensor.shape[i + 1];
    }

    // Calculate strides
    const stride = dim_sizes[positive_axis];

    // Copy data to output tensors
    for (output_shapes, 0..) |shape, out_idx| {
        const split_size = shape[positive_axis];
        const offset = offsets[out_idx];
        const block_size = split_size * stride;

        // Calculate total number of blocks
        var total_blocks: usize = 1;
        for (0..positive_axis) |j| {
            total_blocks *= input_tensor.shape[j];
        }

        // Copy data blocks
        var block_idx: usize = 0;
        while (block_idx < total_blocks) : (block_idx += 1) {
            // Calculate source and destination offsets
            const outer_offset = block_idx * input_tensor.shape[positive_axis] * stride;
            const src_offset = outer_offset + offset * stride;
            const dst_offset = block_idx * split_size * stride;

            // Copy the data block
            @memcpy(output_tensors.*[out_idx].data[dst_offset .. dst_offset + block_size], input_tensor.data[src_offset .. src_offset + block_size]);
        }
    }
}

// Helper to compute offsets for each split
fn compute_split_offsets(input_shape: []const usize, axis: usize, split_sizes: []const usize, num_outputs: usize) ![]usize {
    const dim_size = input_shape[axis];
    var offsets = try pkg_allocator.alloc(usize, num_outputs);
    errdefer pkg_allocator.free(offsets);

    // Calculate offsets based on split sizes
    if (split_sizes.len != num_outputs) return TensorError.InvalidInput;

    var offset: usize = 0;
    for (split_sizes, 0..) |size, i| {
        offsets[i] = offset;
        offset += size;
    }

    if (offset != dim_size) return TensorError.InvalidSplitSize;

    return offsets;
}

pub fn get_split_output_shapes(input_shape: []const usize, axis: i64, split_sizes: ?[]const usize, num_outputs: ?usize) ![][]usize {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_shape.len)) + axis;
        if (adjusted < 0) return TensorError.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_shape.len) return TensorError.InvalidAxis;

    const dim_size = input_shape[positive_axis];
    var sizes = std.ArrayList(usize).init(pkg_allocator);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else if (num_outputs) |n| {
        // Split into equal parts based on the number of outputs
        if (dim_size % n != 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size / n;

        var i: usize = 0;
        while (i < n) : (i += 1) {
            try sizes.append(split_size);
        }
    } else {
        // Default case: just create one output with the full size
        try sizes.append(dim_size);
    }

    // Create output shapes
    var output_shapes = try pkg_allocator.alloc([]usize, sizes.items.len);
    errdefer {
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Fill output shapes
    for (sizes.items, 0..) |split_size, i| {
        output_shapes[i] = try pkg_allocator.alloc(usize, input_shape.len);
        errdefer pkg_allocator.free(output_shapes[i]);

        @memcpy(output_shapes[i], input_shape);
        output_shapes[i][positive_axis] = split_size;
    }

    return output_shapes;
}
