const std = @import("std");
const zant = @import("../../../../zant.zig");
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

pub fn add_bias(comptime T: anytype, tensor: *Tensor(T), bias: *Tensor(T)) !void {
    // Checks:
    if (tensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.shape.len != 1) {
        return TensorMathError.InputTensorsWrongShape;
    }
    const len = bias.shape[0];
    if (len != tensor.shape[tensor.shape.len - 1]) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Instead of using threads, just do it directly
    var index: usize = 0;
    while (index < tensor.size) : (index += len) {
        for (0..len) |i| {
            tensor.data[index + i] += bias.data[i];
        }
    }
}

// Helper function to calculate the broadcasted shape
fn calculate_broadcasted_shape(alloc: *const std.mem.Allocator, shape1_in: []const usize, shape2_in: []const usize) ![]usize {
    const rank1 = shape1_in.len;
    const rank2 = shape2_in.len;
    const max_rank = @max(rank1, rank2);

    // Use temporary allocator for intermediate shapes if needed, actual output shape uses provided allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const tmp_alloc = gpa.allocator();

    // Allocate padded shapes based on max_rank
    const shape1_padded = try tmp_alloc.alloc(usize, max_rank);
    defer tmp_alloc.free(shape1_padded);
    const shape2_padded = try tmp_alloc.alloc(usize, max_rank);
    defer tmp_alloc.free(shape2_padded);

    // Initialize padded shapes with 1s
    @memset(shape1_padded, 1);
    @memset(shape2_padded, 1);

    // Copy original shapes from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1_padded[max_rank - rank1 + i] = shape1_in[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2_padded[max_rank - rank2 + i] = shape2_in[i];
    }

    // Special check: If shape2_in is 1D, try to find a matching dimension in shape1_in
    // This logic needs refinement if we want the bias-like auto-detection.
    // For now, stick to standard broadcasting rules based on padded shapes.
    // TODO: Revisit the bias-like dimension matching logic if needed.

    // Allocate output shape using the main allocator
    const out_shape = try alloc.alloc(usize, max_rank);

    // Verify shapes and calculate output shape
    for (0..max_rank) |dim| {
        if (shape1_padded[dim] != shape2_padded[dim] and shape1_padded[dim] != 1 and shape2_padded[dim] != 1) {
            // Need to free out_shape before returning error
            alloc.free(out_shape);
            std.debug.print("Incompatible broadcast shapes at dim {}: {} vs {}\n", .{ dim, shape1_padded[dim], shape2_padded[dim] }); // DEBUG PRINT
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1_padded[dim], shape2_padded[dim]);
    }

    return out_shape;
}

pub fn sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    // Size check removed here, handled by broadcasting logic or simple case
    // if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize; // Removed check

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    // Create output tensor: Calculate shape *first* if broadcasting might occur
    var out_tensor: Tensor(outputType) = undefined;
    var allocated_shape: ?[]usize = null; // To hold shape if allocated by calculate_broadcasted_shape

    if (std.mem.eql(usize, t1.shape, t2.shape)) {
        // Simple case: shapes are identical
        out_tensor = try Tensor(outputType).fromShape(t1.allocator, t1.shape); // Use t1 or t2 shape
    } else {
        // Broadcasting case: calculate the correct output shape
        const broadcasted_shape = try calculate_broadcasted_shape(t1.allocator, t1.shape, t2.shape);
        // Store the allocated shape so we can free it *later*
        allocated_shape = broadcasted_shape;
        // Tensor.fromShape should copy the shape, but we keep broadcasted_shape alive until after lean_sum_tensors
        out_tensor = try Tensor(outputType).fromShape(t1.allocator, broadcasted_shape);
        // DO NOT free broadcasted_shape here with defer. Free it after lean_sum_tensors.
    }

    try lean_sum_tensors(inputType, outputType, t1, t2, &out_tensor);

    // Free the broadcasted shape *after* lean_sum_tensors is done, if it was allocated.
    if (allocated_shape) |shape_mem| {
        t1.allocator.free(shape_mem);
    }

    return out_tensor;
}

// --------- lean SUM
pub inline fn lean_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    // std.debug.print("\nINFO: Summing tensors with sizes: {d}, {d}\n", .{ t1.size, t2.size }); // DEBUG PRINT
    // std.debug.print("\nINFO: t1 shape: {any}, t2 shape: {any}\n", .{ t1.shape, t2.shape }); // DEBUG PRINT
    // std.debug.print("\nINFO: outputTensor shape: {any}\n", .{outputTensor.shape}); // DEBUG PRINT
    // // Simple case: same size tensors
    if (t1.size == t2.size) {
        // Use unrolled loop for small sizes to avoid SIMD overhead
        if (t1.size <= 8) {
            comptime var unroll = 0;
            inline while (unroll < 8) : (unroll += 1) {
                if (unroll < t1.size and unroll < t2.size) {
                    outputTensor.data[unroll] = @as(outputType, t1.data[unroll] + t2.data[unroll]);
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
                const result = v1 + v2;
                // Use a standard for loop instead of inline while for runtime execution
                for (0..vector_len) |j| {
                    outputTensor.data[i + offset * vector_len + j] = @as(outputType, result[j]);
                }
            }
        }

        // Handle remaining elements with simple loop
        while (i < t1.size) : (i += 1) {
            outputTensor.data[i] = @as(outputType, t1.data[i] + t2.data[i]);
        }
        return;
    }

    // Broadcasting case - use stack arrays for small ranks to avoid allocations
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Use stack arrays for common tensor ranks (up to 4D)
    var stack_shape1: [4]usize = undefined; // Initialize later
    var stack_shape2: [4]usize = undefined;
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

    // Initialize padded shapes with 1s before copying
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy original shape1 from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }

    // Setup shape2: Original logic: copy t2 shape from right-to-left
    i = 0; // Reset i
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // --- START WORKAROUND for potentially truncated outputTensor.shape ---
    // Reconstruct the full output shape based on max_rank, assuming leading 1s were dropped.
    var stack_full_out_shape: [4]usize = undefined;
    const full_output_shape_slice = if (max_rank <= 4) stack_full_out_shape[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    if (max_rank > 4) {
        defer pkg_allocator.free(full_output_shape_slice);
    }

    const output_rank_diff = max_rank - outputTensor.shape.len;
    if (output_rank_diff > 0) {
        @memset(full_output_shape_slice[0..output_rank_diff], 1); // Pad with leading 1s
    }
    @memcpy(full_output_shape_slice[output_rank_diff..], outputTensor.shape);
    // std.debug.print("DEBUG: Original output shape: {any}, Reconstructed full shape: {any}\\n", .{ outputTensor.shape, full_output_shape_slice });
    // --- END WORKAROUND ---

    // Calculate strides from right to left using the reconstructed full shape
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        // Use the reconstructed shape here
        stride *= full_output_shape_slice[i]; // Use reconstructed shape
    }

    // Perform addition with broadcasting
    // Use stack arrays for common tensor ranks (up to 4D) - indices were already allocated above
    var stack_loop_indices: [4]usize = [_]usize{0} ** 4;
    const loop_indices = if (max_rank <= 4) stack_loop_indices[0..max_rank] else indices; // Reuse allocated 'indices' if max_rank > 4

    // Initialize loop_indices if using the stack allocation
    if (max_rank <= 4) {
        @memset(loop_indices, 0);
    }

    i = 0; // Reset i before the loop, don't redeclare
    while (i < outputTensor.size) : (i += 1) {
        // Calculate multi-dimensional indices for current output position 'i'
        var temp = i;
        for (0..max_rank) |dim| {
            const idx = max_rank - 1 - dim; // Iterate dimensions from right-to-left
            loop_indices[idx] = temp / out_strides[idx];
            temp = temp % out_strides[idx];
        }

        // Calculate linear input indices (idx1, idx2) using multi-dimensional indices and strides
        var idx1: usize = 0;
        var idx2: usize = 0;
        for (0..max_rank) |dim| {
            // stridesN[dim] is 0 if shapeN[dim] is 1, handling broadcasting implicitly
            idx1 += loop_indices[dim] * strides1[dim];
            idx2 += loop_indices[dim] * strides2[dim];
        }

        // Perform the addition
        outputTensor.data[i] = t1.data[idx1] + t2.data[idx2];
    }
}

/// Returns a Tensor with the same shape as the input tensors, where each element is the sum of all tensors at that location
pub fn sum_tensor_list(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType)) !Tensor(outputType) {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;
    if (tensors.len == 1) {
        var out_tensor = try Tensor(outputType).fromShape(tensors[0].allocator, tensors[0].shape);
        for (0..tensors[0].data.len) |i| {
            out_tensor.data[i] = tensors[0].data[i];
        }
        return out_tensor;
    }

    // Use first tensor as reference for size and shape checks
    const ref_tensor = tensors[0];

    // Check all tensors have same size
    for (tensors[1..]) |t| {
        if (t.size != ref_tensor.size) return TensorMathError.InputTensorDifferentSize;
    }

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    var out_tensor = try Tensor(outputType).fromShape(ref_tensor.allocator, ref_tensor.shape);
    try lean_sum_tensor_list(inputType, outputType, tensors, &out_tensor);

    return out_tensor;
}

pub inline fn lean_sum_tensor_list(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Initialize output with first tensor
    for (0..tensors[0].data.len) |i| {
        outputTensor.data[i] = tensors[0].data[i];
    }

    // Add remaining tensors
    for (tensors[1..]) |t| {
        for (0..t.data.len) |i| {
            outputTensor.data[i] += t.data[i];
        }
    }
}

/// Lower an ONNX "Add" with NumPy-style broadcasting into UOps,
/// **without** emitting a final FUSE hint.
pub fn lowerAdd(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    B_id: usize,
    out_shape: []const usize, // broadcasted shape
    strideA: []const isize, // per-dim strides (0 ⇒ broadcast)
    strideB: []const isize,
    out_dtype: DType, // promoted element type
) usize { // returns id of result buffer
    // ── Set-up phase ────────────────────────────────────────────────────
    // _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)
    // _ = b.push(.SHAPE, .i32, &.{B_id}, null); // b_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

    const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideB } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ───────────────────────────────────────────────
    var nelem: usize = 1;
    for (out_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, id_range }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
    const id_loadB = b.push(.LOAD, out_dtype, &.{id_gepB}, null);

    const id_add = b.push(.ADD, out_dtype, &.{ id_loadA, id_loadB }, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_add }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf; // SSA id of the output tensor
}
