const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

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
pub fn calculate_broadcasted_shape(alloc: *const std.mem.Allocator, shape1_in: []const usize, shape2_in: []const usize) ![]usize {
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
            // std.log.warn("Incompatible broadcast shapes at dim {}: {} vs {}\n", .{ dim, shape1_padded[dim], shape2_padded[dim] }); // DEBUG PRINT
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1_padded[dim], shape2_padded[dim]);
    }

    return out_shape;
}
