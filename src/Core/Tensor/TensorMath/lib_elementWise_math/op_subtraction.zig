const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;

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
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // Verify shapes are compatible for broadcasting
    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Calculate total size and strides
    var total_size: usize = 1;
    var strides1 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(strides1);
    var strides2 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(strides2);
    var out_strides = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(out_strides);

    // Calculate strides from right to left
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        stride *= out_shape[i];
    }
    total_size = stride;

    std.debug.print("\nshape1: {any}, strides1: {any}", .{ shape1, strides1 });
    std.debug.print("\nshape2: {any}, strides2: {any}", .{ shape2, strides2 });
    std.debug.print("\nout_shape: {any}, out_strides: {any}", .{ out_shape, out_strides });

    // Allocate output tensor
    var out_data = try pkg_allocator.alloc(outputType, total_size);
    errdefer pkg_allocator.free(out_data);

    // Perform subtraction with broadcasting
    var indices = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(indices);
    @memset(indices, 0);

    i = 0;
    while (i < total_size) : (i += 1) {
        // Calculate indices for current position
        var temp = i;
        for (0..max_rank) |dim| {
            const idx = max_rank - 1 - dim;
            indices[idx] = temp / out_strides[idx];
            temp = temp % out_strides[idx];
        }

        // Calculate input indices considering broadcasting
        var idx1: usize = 0;
        var idx2: usize = 0;

        // For same shape tensors, use the same index calculation
        if (std.mem.eql(usize, shape1, shape2)) {
            idx1 = i;
            idx2 = i;
        } else {
            // For broadcasting case
            for (0..max_rank) |dim| {
                const pos = indices[dim];
                // For t1: if dimension is 1, don't increment index (broadcasting)
                if (shape1[dim] > 1) {
                    idx1 += pos * strides1[dim];
                }
                // For t2: if dimension is 1, don't increment index (broadcasting)
                if (shape2[dim] > 1) {
                    const t2_pos = pos % shape2[dim];
                    idx2 += t2_pos * strides2[dim];
                }
            }
        }

        // Perform subtraction: t1 - t2
        const val1 = t1.data[idx1];
        const val2 = t2.data[idx2];
        out_data[i] = val1 - val2;
        std.debug.print("\nFinal: idx1={}, val1={}, idx2={}, val2={}, result={}", .{ idx1, val1, idx2, val2, out_data[i] });
    }

    return Tensor(outputType){
        .data = out_data,
        .shape = out_shape,
        .size = total_size,
        .allocator = &pkg_allocator,
        .owns_memory = true,
    };
}
