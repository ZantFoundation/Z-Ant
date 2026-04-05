const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

/// Performs Element-wise binary division of two tensors with support for broadcasting.
pub fn div(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T)) !Tensor(T) {
    // Handle broadcasting
    const rank1 = lhs.shape.len;
    const rank2 = rhs.shape.len;
    const max_rank = @max(rank1, rank2);

    // Create output tensor with broadcasted shape
    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    // Pad shapes with 1s for broadcasting
    var shape1 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape1);
    var shape2 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape2);

    // Initialize with 1s
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy original shapes from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = lhs.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = rhs.shape[i];
    }

    // Calculate broadcasted shape
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return error.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Create output tensor
    var out_tensor = try Tensor(T).fromShape(lhs.allocator, out_shape);
    errdefer out_tensor.deinit();
    pkg_allocator.free(out_shape); // Free out_shape after creating tensor

    try div_lean(T, lhs, rhs, &out_tensor);

    return out_tensor;
}

// --------- lean DIV
pub inline fn div_lean(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T), result: *Tensor(T)) !void {

    // Simple case: same size tensors
    if (lhs.size == rhs.size and std.mem.eql(usize, lhs.shape, rhs.shape)) {
        for (0..lhs.size) |i| {
            result.data[i] = lhs.data[i] / rhs.data[i];
        }
        return;
    }

    // Broadcasting case - use stack arrays for small ranks to avoid allocations
    const rank1 = lhs.shape.len;
    const rank2 = rhs.shape.len;
    const max_rank = @max(rank1, rank2);

    // Use stack arrays for common tensor ranks (up to 4D)
    var stack_shape1: [4]usize = [_]usize{1} ** 4;
    var stack_shape2: [4]usize = [_]usize{1} ** 4;
    var stack_strides1: [4]usize = undefined;
    var stack_strides2: [4]usize = undefined;

    const shape1 = if (max_rank <= 4) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const shape2 = if (max_rank <= 4) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides1 = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides2 = if (max_rank <= 4) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

    // Free heap-allocated arrays at block scope exit (only if actually heap-allocated)
    defer if (max_rank > 4) pkg_allocator.free(shape1);
    defer if (max_rank > 4) pkg_allocator.free(shape2);
    defer if (max_rank > 4) pkg_allocator.free(strides1);
    defer if (max_rank > 4) pkg_allocator.free(strides2);

    // Copy original shapes from right to left (align trailing dimensions)
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = lhs.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = rhs.shape[i];
    }

    // Verify shapes are compatible for broadcasting
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return error.IncompatibleBroadcastShapes;
        }
    }

    // Calculate strides for each input tensor based on their ORIGINAL shapes, not the broadcasted shapes
    // For tensor 1 (lhs)
    var stride: usize = 1;
    i = rank1;
    while (i > 0) {
        i -= 1;
        const padded_dim = max_rank - rank1 + i;
        strides1[padded_dim] = stride;
        stride *= lhs.shape[i];
    }
    // Fill leading dimensions with 0 stride (broadcasting)
    for (0..(max_rank - rank1)) |dim| {
        strides1[dim] = 0;
    }

    // For tensor 2 (rhs)
    stride = 1;
    i = rank2;
    while (i > 0) {
        i -= 1;
        const padded_dim = max_rank - rank2 + i;
        strides2[padded_dim] = stride;
        stride *= rhs.shape[i];
    }
    // Fill leading dimensions with 0 stride (broadcasting)
    for (0..(max_rank - rank2)) |dim| {
        strides2[dim] = 0;
    }

    // Override strides for dimensions of size 1 (these should broadcast)
    for (0..max_rank) |dim| {
        if (shape1[dim] == 1) {
            strides1[dim] = 0;
        }
        if (shape2[dim] == 1) {
            strides2[dim] = 0;
        }
    }

    // Calculate output strides
    stride = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        stride *= result.shape[i];
    }

    // Calculate output strides for coordinate conversion
    var out_strides = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    defer if (max_rank > 4) pkg_allocator.free(out_strides);

    stride = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        stride *= result.shape[i];
    }

    // Perform division with broadcasting
    for (0..result.size) |linear_idx| {
        // Convert linear index to multi-dimensional coordinates
        var idx1: usize = 0;
        var idx2: usize = 0;

        // Calculate coordinates and map to input indices
        for (0..max_rank) |dim| {
            const coord = (linear_idx / out_strides[dim]) % result.shape[dim];

            // Map coordinate to input tensor indices with broadcasting
            const coord1 = if (shape1[dim] == 1) 0 else coord;
            const coord2 = if (shape2[dim] == 1) 0 else coord;

            idx1 += coord1 * strides1[dim];
            idx2 += coord2 * strides2[dim];
        }

        result.data[linear_idx] = lhs.data[idx1] / rhs.data[idx2];
    }
}
