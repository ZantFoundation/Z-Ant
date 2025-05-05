const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

const pkg_allocator = zant.utils.allocator.allocator;

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  flip_matrix( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn flip_matrix(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    try flip_matrix_lean(T, kernel, &flipped_kernel);
    return flipped_kernel;
}

pub fn flip_matrix_lean(comptime T: type, kernel: *Tensor(T), output_kernel: *Tensor(T)) !void {
    const kernel_dim = kernel.shape.len;
    const kernel_row = kernel.shape[kernel_dim - 2];
    const kernel_cols = kernel.shape[kernel_dim - 1];
    const matrix_dim = kernel_cols * kernel_row;

    const total_number_2DMatrices = output_kernel.size / matrix_dim;

    for (0..total_number_2DMatrices) |matix_i| {
        for (0..kernel_row) |i| {
            for (0..kernel_cols) |j| {
                output_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = kernel.data[matix_i * matrix_dim + i * kernel_cols + j];
            }
        }
    }
}

/// Computes element-wise negation, multiplying each element by -1
/// This is the ONNX Neg operation: Y = -X
pub fn neg(comptime T: type, tensor: *Tensor(T)) !Tensor(T) {
    const neg_shape = get_neg_output_shape(tensor.shape);
    defer tensor.allocator.free(neg_shape);

    var neg_tensor = try Tensor(T).fromShape(tensor.allocator, neg_shape);

    try neg_lean(T, tensor, &neg_tensor);
    return neg_tensor;
}

/// Element-wise negation implementation (multiplies each element by -1)
pub fn neg_lean(comptime T: type, input: *Tensor(T), output: *Tensor(T)) !void {
    if (output.size != input.size) {
        return TensorError.MismatchedShape;
    }

    for (0..input.size) |i| {
        output.data[i] = -input.data[i];
    }
}

pub fn get_neg_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

// https://onnx.ai/onnx/operators/onnx__Neg.html
pub fn lowerNeg(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    strideA: []const isize, // per-dim strides (0 ⇒ broadcast)
    out_shape: []const usize,
    out_dtype: DType, // promoted element type
) usize { // returns id of result buffer

    // ── Set-up phase ────────────────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ────────────────────────────────────────────────

    var nelem: usize = 1;
    for (out_shape) |dim| nelem *= dim;

    const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

    const id_neg = b.push(.NEG, out_dtype, &.{id_loadA}, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_neg }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf;
}
