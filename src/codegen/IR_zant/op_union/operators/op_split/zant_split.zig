const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

const get_split_output_shapes = @import("utils_split.zig").get_split_output_shapes;

/// Split a tensor into multiple tensors along a specified axis.
/// If split_sizes is null, the tensor is split into equal parts.
/// If split_sizes is provided, it specifies the size of each split.
/// Negative axis values count from the back (-1 means last axis).
/// Returns an array of tensors that must be freed by the caller.
pub fn split(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const i64) ![]Tensor(T) {
    // Handle negative axis
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    if (positive_axis >= t.shape.len) return error.InvalidAxis;

    // Calculate split sizes
    const dim_size = t.shape[positive_axis];
    var sizes: std.ArrayList(i64) = .empty;
    defer sizes.deinit(t.allocator.*);

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: i64 = 0;
        for (s) |size| {
            if (size < 0) return error.InvalidSplitSize;
            try sizes.append(t.allocator.*, size);
            total_size += size;
        }
        if (total_size != dim_size) return error.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return error.InvalidSplitSize;
        const split_size = @as(i64, @intCast(dim_size));
        try sizes.append(t.allocator.*, split_size);
    }

    const output_shapes = try get_split_output_shapes(t.shape, axis, sizes.items, sizes.items.len);
    defer {
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Create output tensors with fully allocated buffers so `split_lean`
    // can focus on validation and data movement only.
    var output_tensors = try t.allocator.alloc(Tensor(T), output_shapes.len);
    var initialized_outputs: usize = 0;
    errdefer {
        for (output_tensors[0..initialized_outputs]) |*tensor| {
            tensor.deinit();
        }
        t.allocator.free(output_tensors);
    }

    for (output_shapes, 0..) |shape, i| {
        output_tensors[i] = try Tensor(T).fromShape(t.allocator, shape);
        initialized_outputs += 1;
    }

    // Create a durable copy of the split sizes using the ONNX-facing i64 type.
    const durable_split_sizes = try t.allocator.dupe(i64, sizes.items);
    defer t.allocator.free(durable_split_sizes);

    try split_lean(T, t, axis, durable_split_sizes, &output_tensors);

    return output_tensors;
}

//lean split
//inputs:
//split_sizes can't be null
pub fn split_lean(comptime T: type, input_tensor: *Tensor(T), axis: i64, split_sizes: []const i64, output_tensors: *[]Tensor(T)) !void {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_tensor.shape.len)) + axis;
        if (adjusted < 0) return error.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_tensor.shape.len) return error.InvalidAxis;

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
        return error.InvalidInput;
    }

    for (output_shapes, 0..) |shape, i| {
        // Calculate required size
        var total_size: usize = 1;
        for (shape) |dim| total_size *= dim;

        const output_tensor = &output_tensors.*[i];
        if (output_tensor.shape.len != shape.len) return error.InvalidInput;
        if (!std.mem.eql(usize, output_tensor.shape, shape)) return error.InvalidInput;
        if (output_tensor.data.len != total_size or output_tensor.size != total_size) return error.InvalidInput;
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
        const offset: usize = @intCast(offsets[out_idx]);
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
fn compute_split_offsets(input_shape: []const usize, axis: usize, split_sizes: []const i64, num_outputs: usize) ![]i64 {
    const dim_size: i64 = @intCast(input_shape[axis]);
    var offsets = try pkg_allocator.alloc(i64, @intCast(num_outputs));
    errdefer pkg_allocator.free(offsets);

    // Calculate offsets based on split sizes
    if (split_sizes.len != num_outputs) return error.InvalidInput;

    var offset: i64 = 0;
    for (split_sizes, 0..) |size, i| {
        if (size < 0) return error.InvalidSplitSize;
        offsets[i] = offset;
        offset += size;
    }

    if (offset != dim_size) return error.InvalidSplitSize;

    return offsets;
}
