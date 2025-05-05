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

    // Calculate broadcasted shape
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Calculate total size
    var total_size: usize = 1;
    for (0..max_rank) |dim| {
        total_size *= out_shape[dim];
    }

    // Create output tensor
    var out_tensor = try Tensor(outputType).fromShape(&pkg_allocator, out_shape);
    errdefer out_tensor.deinit();
    pkg_allocator.free(out_shape); // Free out_shape after creating tensor

    try lean_sub_tensors(inputType, outputType, t1, t2, &out_tensor);

    return out_tensor;
}

pub inline fn lean_sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    // Simple case: same size tensors
    if (t1.size == t2.size) {
        // Use unrolled loop for small sizes to avoid SIMD overhead
        if (t1.size <= 8) {
            comptime var unroll = 0;
            inline while (unroll < 8) : (unroll += 1) {
                if (unroll < t1.size) {
                    outputTensor.data[unroll] = @as(outputType, t1.data[unroll] - t2.data[unroll]);
                }
            }
            return;
        }

        // Use SIMD for larger sizes
        const vector_len = std.simd.suggestVectorLength(inputType) orelse 4;
        const Vec = @Vector(vector_len, inputType);

        // Process 4 vectors at once to exploit instruction-level parallelism
        const chunk_size = vector_len * 4;
        const chunks = t1.size / chunk_size;
        var i: usize = 0;

        while (i < chunks * chunk_size) : (i += chunk_size) {
            inline for (0..4) |offset| {
                const v1: Vec = t1.data[i + offset * vector_len ..][0..vector_len].*;
                const v2: Vec = t2.data[i + offset * vector_len ..][0..vector_len].*;
                const result = v1 - v2;
                comptime var j = 0;
                inline while (j < vector_len) : (j += 1) {
                    outputTensor.data[i + offset * vector_len + j] = @as(outputType, result[j]);
                }
            }
        }

        // Handle remaining elements with simple loop
        while (i < t1.size) : (i += 1) {
            outputTensor.data[i] = @as(outputType, t1.data[i] - t2.data[i]);
        }
        return;
    }

    // Broadcasting case - use stack arrays for small ranks to avoid allocations
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Use stack arrays for common tensor ranks (up to 4D)
    var stack_shape1: [4]usize = [_]usize{1} ** 4;
    var stack_shape2: [4]usize = [_]usize{1} ** 4;
    var stack_strides1: [4]usize = undefined;
    var stack_strides2: [4]usize = undefined;
    var stack_out_strides: [4]usize = undefined;
    var stack_indices: [4]usize = [_]usize{0} ** 4;

    const shape1 = if (max_rank <= 4) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const shape2 = if (max_rank <= 4) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides1 = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides2 = if (max_rank <= 4) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const out_strides = if (max_rank <= 4) stack_out_strides[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const indices = if (max_rank <= 4) stack_indices[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

    // Only defer if we actually allocated
    if (max_rank > 4) {
        defer pkg_allocator.free(shape1);
        defer pkg_allocator.free(shape2);
        defer pkg_allocator.free(strides1);
        defer pkg_allocator.free(strides2);
        defer pkg_allocator.free(out_strides);
        defer pkg_allocator.free(indices);
    }

    // Copy original shapes from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // Calculate strides from right to left
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        stride *= outputTensor.shape[i];
    }

    // Perform subtraction with broadcasting
    @memset(indices, 0);

    i = 0;
    while (i < outputTensor.size) : (i += 1) {
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

        outputTensor.data[i] = t1.data[idx1] - t2.data[idx2];
    }
}

/// https://onnx.ai/onnx/operators/onnx__Sub.html
pub fn lowerSub(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    B_id: usize,
    out_shape: []const usize, // broadcasted shape
    strideA: []const isize, // per-dim strides (0 ⇒ broadcast)
    strideB: []const isize,
    out_dtype: DType, // promoted element type
) usize { // returns id of result buffer

    // ── Set-up phase ────────────────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)
    _ = b.push(.SHAPE, .i32, &.{B_id}, null); // b_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

    const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideB } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ───────────────────────────────────────────────
    var nelem: usize = 1;
    for (out_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, id_range }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
    const id_loadB = b.push(.LOAD, out_dtype, &.{id_gepB}, null);

    const id_sub = b.push(.SUB, out_dtype, &.{ id_loadA, id_loadB }, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_sub }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}
