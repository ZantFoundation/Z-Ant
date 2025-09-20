const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

/// Performs element-wise binary subtraction with Numpy-style broadcasting support
pub fn sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // If input and output are same type, no check needed
    } else {
        if (@bitSizeOf(outputType) <= 16) { //quantized
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    // Handle broadcasting
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Create output tensor with broadcasted shape
    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    // Pad shapes with 1s for broadcasting (NumPy rule: pad on the left/leading side)
    var shape1 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape1);
    var shape2 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape2);

    // Initialize with 1s
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy original shapes from right to left (trailing alignment)
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // Calculate broadcasted shape (NumPy broadcasting rules)
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Create output tensor
    var out_tensor = try Tensor(outputType).fromShape(&pkg_allocator, out_shape);
    errdefer out_tensor.deinit();
    pkg_allocator.free(out_shape); // Free out_shape after creating tensor

    try lean_sub_tensors(inputType, outputType, t1, t2, &out_tensor);

    return out_tensor;
}

// Helper for mixed precision subtraction
pub inline fn lean_sub_tensors_mixed(comptime T1: type, comptime T2: type, comptime outputType: anytype, t1: *Tensor(T1), t2: *Tensor(T2), outputTensor: *Tensor(outputType)) !void {
    if (T1 == T2) {
        // Same types - use the regular function
        return lean_sub_tensors(T1, outputType, t1, @ptrCast(t2), outputTensor);
    } else {
        // Different types - convert both to output type for computation
        const rank1 = t1.shape.len;
        const rank2 = t2.shape.len;

        // Fast path: identical shapes, no broadcasting needed
        if (rank1 == rank2 and std.mem.eql(usize, t1.shape, t2.shape)) {
            for (0..t1.size) |i| {
                const val1: outputType = if (T1 == outputType) t1.data[i] else @floatCast(t1.data[i]);
                const val2: outputType = if (T2 == outputType) t2.data[i] else @floatCast(t2.data[i]);
                outputTensor.data[i] = val1 - val2;
            }
        } else {
            // TODO: implement broadcasting for mixed types
            return TensorMathError.BroadcastingNotSupportedMixed;
        }
    }
}

pub inline fn lean_sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Fast path: identical shapes, no broadcasting needed
    if (rank1 == rank2 and std.mem.eql(usize, t1.shape, t2.shape)) {
        // Use SIMD for larger sizes when possible
        const vector_len: usize = std.simd.suggestVectorLength(inputType) orelse 4;
        const Vec = @Vector(vector_len, inputType);

        if (t1.size >= vector_len) {
            // Process data in SIMD-sized chunks
            const aligned_len = (t1.size / vector_len) * vector_len;
            var i: usize = 0;

            // Main SIMD loop
            while (i < aligned_len) : (i += vector_len) {
                const v1: Vec = t1.data[i..][0..vector_len].*;
                const v2: Vec = t2.data[i..][0..vector_len].*;
                const result = v1 - v2;

                // Store results
                for (0..vector_len) |j| {
                    outputTensor.data[i + j] = @as(outputType, result[j]);
                }
            }

            // Handle remaining elements
            while (i < t1.size) : (i += 1) {
                outputTensor.data[i] = @as(outputType, t1.data[i] - t2.data[i]);
            }
            return;
        } else {
            var i: usize = 0;
            while (i < t1.size) : (i += 1) {
                outputTensor.data[i] = @as(outputType, t1.data[i] - t2.data[i]);
            }
            return;
        }
    }

    // Broadcasting path - this is where the main fix is needed
    var stack_shape1: [8]usize = [_]usize{1} ** 8;
    var stack_shape2: [8]usize = [_]usize{1} ** 8;

    // Use stack for common ranks, heap for larger
    const shape1 = if (max_rank <= 8) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const shape2 = if (max_rank <= 8) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

    if (max_rank > 8) {
        defer {
            pkg_allocator.free(shape1);
            pkg_allocator.free(shape2);
        }
    }

    // Prepare shapes for broadcasting (pad on the left with 1s)
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy shapes from right to left (trailing alignment)
    for (0..rank1) |i| {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    for (0..rank2) |i| {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }


    // Calculate original strides for both input tensors
    var stack_strides1: [8]usize = undefined;
    var stack_strides2: [8]usize = undefined;
    const strides1 = if (rank1 <= 8) stack_strides1[0..rank1] else try pkg_allocator.alloc(usize, rank1);
    const strides2 = if (rank2 <= 8) stack_strides2[0..rank2] else try pkg_allocator.alloc(usize, rank2);

    if (rank1 > 8 or rank2 > 8) {
        defer {
            if (rank1 > 8) pkg_allocator.free(strides1);
            if (rank2 > 8) pkg_allocator.free(strides2);
        }
    }

    // Calculate strides for t1
    var stride: usize = 1;
    var dim = rank1;
    while (dim > 0) {
        dim -= 1;
        strides1[dim] = stride;
        stride *= t1.shape[dim];
    }

    // Calculate strides for t2
    stride = 1;
    dim = rank2;
    while (dim > 0) {
        dim -= 1;
        strides2[dim] = stride;
        stride *= t2.shape[dim];
    }


    // Build full output shape from broadcasted shapes
    var stack_full_out_shape: [8]usize = undefined;
    const full_out_shape = stack_full_out_shape[0..max_rank];
    for (0..max_rank) |d| {
        full_out_shape[d] = @max(shape1[d], shape2[d]);
    }


    // Output strides and loop indices
    var stack_out_strides: [8]usize = undefined;
    var stack_indices: [8]usize = [_]usize{0} ** 8;
    const out_strides = stack_out_strides[0..max_rank];
    const loop_indices = stack_indices[0..max_rank];

    // Broadcasted strides across max_rank (0 for broadcasted dims)
    var stack_bcast_strides1: [8]usize = undefined;
    var stack_bcast_strides2: [8]usize = undefined;
    const bcast_strides1 = stack_bcast_strides1[0..max_rank];
    const bcast_strides2 = stack_bcast_strides2[0..max_rank];

    // Compute out strides and broadcasted input strides from right to left
    var out_stride: usize = 1;
    var actual_stride1: usize = 1;
    var actual_stride2: usize = 1;
    var i_rev: usize = max_rank;
    while (i_rev > 0) {
        i_rev -= 1;
        out_strides[i_rev] = out_stride;
        // For broadcasted strides, only advance stride when the dimension is > 1
        bcast_strides1[i_rev] = if (shape1[i_rev] > 1) actual_stride1 else 0;
        bcast_strides2[i_rev] = if (shape2[i_rev] > 1) actual_stride2 else 0;

        out_stride *= full_out_shape[i_rev];
        actual_stride1 *= shape1[i_rev];
        actual_stride2 *= shape2[i_rev];
    }


    // Iterate over the output tensor linearly, map to multi-d indices via out_strides
    for (0..outputTensor.size) |flat_idx| {
        var remaining = flat_idx;
        for (0..max_rank) |d| {
            const s = out_strides[d];
            loop_indices[d] = @divFloor(remaining, s);
            remaining = @mod(remaining, s);
        }

        // Compute input linear indices using effective strides
        var idx1: usize = 0;
        var idx2: usize = 0;
        for (0..max_rank) |d| {
            if (bcast_strides1[d] != 0) {
                idx1 += (@mod(loop_indices[d], shape1[d])) * bcast_strides1[d];
            }
            if (bcast_strides2[d] != 0) {
                idx2 += (@mod(loop_indices[d], shape2[d])) * bcast_strides2[d];
            }
        }

        if (idx1 >= t1.size or idx2 >= t2.size) {
            @panic("sub_tensors: index out of bounds");
        }

        outputTensor.data[flat_idx] = @as(outputType, t1.data[idx1] - t2.data[idx2]);
    }
}
