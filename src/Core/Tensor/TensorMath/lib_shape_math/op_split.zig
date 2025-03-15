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

    const split_size = sizes.items;
    try split_lean(T, t, axis, split_size, &output_tensors);

    return output_tensors;
}

//lean split
//inputs:
//split_sizes can't be null
pub fn split_lean(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const usize, output_tensors: *[]Tensor(T)) !void {
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    const dim_size = t.shape[positive_axis];
    const sizes = split_sizes.?;

    var offset: usize = 0;
    for (sizes, 0..) |split_size, i| {
        // Create shape for the split tensor
        var new_shape = try t.allocator.alloc(usize, t.shape.len);
        errdefer t.allocator.free(new_shape);
        @memcpy(new_shape, t.shape);
        new_shape[positive_axis] = split_size;

        // Calculate total size for the split tensor
        var total_size: usize = 1;
        for (new_shape) |dim| {
            total_size *= dim;
        }

        // Allocate memory for the split tensor's data
        var new_data = try t.allocator.alloc(T, total_size);
        errdefer t.allocator.free(new_data);

        // Calculate strides
        var stride: usize = 1;
        for (positive_axis + 1..t.shape.len) |j| {
            stride *= t.shape[j];
        }

        // Copy data to the split tensor
        const block_size = split_size * stride;
        const num_blocks = total_size / block_size;

        var block_idx: usize = 0;
        while (block_idx < num_blocks) : (block_idx += 1) {
            const src_start = offset + block_idx * dim_size * stride;
            const dst_start = block_idx * split_size * stride;
            const copy_size = split_size * stride;
            @memcpy(new_data[dst_start .. dst_start + copy_size], t.data[src_start .. src_start + copy_size]);
        }

        // Create the split tensor
        output_tensors.*[i].data = new_data;
        output_tensors.*[i].size = total_size;
        output_tensors.*[i].shape = new_shape;
        output_tensors.*[i].allocator = t.allocator;
        output_tensors.*[i].owns_memory = true;

        offset += split_size * stride;
    }
}

pub fn get_split_output_shapes(input_shape: []const usize, axis: i64, split_sizes: ?[]const usize) ![][]usize {
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

    // Rest of the function remains the same...
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
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
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
