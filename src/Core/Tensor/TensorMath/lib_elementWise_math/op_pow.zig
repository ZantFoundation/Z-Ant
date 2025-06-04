const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorError = error_handler.TensorError;
const TensorMathError = error_handler.TensorMathError;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

// --------- standard POW
pub fn pow(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T)) !Tensor(T) {
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
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Create output tensor
    var out_tensor = try Tensor(T).fromShape(lhs.allocator, out_shape);
    errdefer out_tensor.deinit();
    pkg_allocator.free(out_shape); // Free out_shape after creating tensor

    try pow_lean(T, lhs, rhs, &out_tensor);

    return out_tensor;
}

// --------- lean POW
pub inline fn pow_lean(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T), result: *Tensor(T)) !void {
    // Simple case: same size tensors
    if (lhs.size == rhs.size and std.mem.eql(usize, lhs.shape, result.shape)) {
        for (0..lhs.size) |i| {
            result.data[i] = pow_elem(T, lhs.data[i], rhs.data[i]);
        }
        return;
    }

    // Broadcasting case - use stack arrays for small ranks to avoid allocations
    const rank1 = lhs.shape.len;
    const rank2 = rhs.shape.len;
    const max_rank = @max(rank1, rank2);

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
        shape1[max_rank - rank1 + i] = lhs.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = rhs.shape[i];
    }

    // Calculate strides from right to left
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        const out_dim_size = @max(shape1[i], shape2[i]);
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        stride *= out_dim_size;
    }

    @memset(indices, 0);

    i = 0;
    while (i < result.size) : (i += 1) {
        var temp = i;
        for (0..max_rank) |dim| {
            const idx = max_rank - 1 - dim;
            indices[idx] = temp / out_strides[idx];
            temp = temp % out_strides[idx];
        }

        var idx1: usize = 0;
        var idx2: usize = 0;

        if (std.mem.eql(usize, shape1, shape2)) {
            idx1 = i;
            idx2 = i;
        } else {
            for (0..max_rank) |dim| {
                const pos = indices[dim];
                if (shape1[dim] > 1) {
                    idx1 += pos * strides1[dim];
                }
                if (shape2[dim] > 1) {
                    const t2_pos = pos % shape2[dim];
                    idx2 += t2_pos * strides2[dim];
                }
            }
        }

        result.data[i] = pow_elem(T, lhs.data[idx1], rhs.data[idx2]);
    }
}

// Helper for pow elementwise, handles float/int
fn pow_elem(comptime T: type, x: T, y: T) T {
    switch (@typeInfo(T)) {
        .float => return std.math.pow(T, x, y),
        .int => return std.math.powi(T, x, y),
        else => @compileError("Unsupported type for pow_elem"),
    }
}

// Calculate the output shape for element-wise pow with broadcasting.
pub fn get_pow_output_shape(lhs: []const usize, rhs: []const usize) ![]usize {
    const rank1 = lhs.len;
    const rank2 = rhs.len;
    const max_rank = @max(rank1, rank2);

    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    var i: usize = 0;
    while (i < max_rank) : (i += 1) {
        const dim1 = if (rank1 + i >= max_rank) lhs[rank1 + i - max_rank] else 1;
        const dim2 = if (rank2 + i >= max_rank) rhs[rank2 + i - max_rank] else 1;

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            std.log.warn("Incompatible broadcast shapes: dim1={}, dim2={}, index={}\n", .{ dim1, dim2, i });
            std.log.warn("lhs shape: {any}\n", .{lhs});
            std.log.warn("rhs shape: {any}\n", .{rhs});
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[i] = @max(dim1, dim2);
    }

    return out_shape;
}

/// https://onnx.ai/onnx/operators/onnx__Pow.html
pub fn lowerPow(
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

    const id_pow = b.push(.POW, out_dtype, &.{ id_loadA, id_loadB }, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_pow }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
} 