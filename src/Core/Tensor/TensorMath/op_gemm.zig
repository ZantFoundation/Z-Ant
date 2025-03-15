const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const TensMath = @import("tensor_math_standard.zig");
const op_mat_mul = @import("op_mat_mul.zig");

// Note that this function cuold benefit from SIMD optimizations

/// Implements the GEMM operator from the ONNX standard https://onnx.ai/onnx/operators/onnx__Gemm.html Y = alpha*A*B + beta*C
/// NOTE: (IMPORTANT FOR CODE GEN) Since multibatch/multichannel is not supported by mat_mul neither gemm does. Remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
/// The broadcasting only accours in rows and cols dimensions, while batch and channel dimensions must have the same size between the operands.
pub fn gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: f32, beta: f32, transA: bool, transB: bool) !Tensor(T) {
    var cond_A: usize = 0;
    var cond_B: usize = 0;
    var res_rows: usize = 0;
    var res_cols: usize = 0;

    // verifying correct shape, batch, channels
    if (A.shape.len != 4 or B.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }
    if (A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]) {
        return TensorMathError.InputTensorDifferentShape;
    }

    // verifying matrix multiplication viability, getting result shape
    if (transA) {
        cond_A = A.shape[2];
        res_rows = A.shape[3];
    } else {
        cond_A = A.shape[3];
        res_rows = A.shape[2];
    }

    if (transB) {
        cond_B = B.shape[3];
        res_cols = B.shape[2];
    } else {
        cond_B = B.shape[2];
        res_cols = B.shape[3];
    }

    if (cond_A != cond_B) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // control on optional tensor C unwrapping it
    if (C) |actual_C| {
        if (actual_C.shape.len != 4) {
            return TensorMathError.InputTensorsWrongShape;
        }
        if (actual_C.shape[0] != A.shape[0] or actual_C.shape[1] != A.shape[1]) {
            return TensorMathError.InputTensorDifferentShape;
        }
        if ((actual_C.shape[2] != res_rows and actual_C.shape[2] != 1) or (actual_C.shape[3] != res_cols and actual_C.shape[3] != 1)) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
    }

    // creating result tensor
    var res_shape = try pkg_allocator.alloc(usize, 4);
    defer pkg_allocator.free(res_shape);
    res_shape[0] = A.shape[0];
    res_shape[1] = A.shape[1];
    res_shape[2] = res_rows;
    res_shape[3] = res_cols;

    var result = try Tensor(T).fromShape(&pkg_allocator, res_shape);
    errdefer result.deinit();

    // debug
    // for (0..A.data.len) |i|
    //     std.debug.print("\n gemm A[{d}] {d}", .{ i, A.data[i] });
    // for (0..B.data.len) |i|
    //     std.debug.print("\n gemm B[{d}] {d}", .{ i, B.data[i] });
    // if (C) |CC| {
    //     for (0..CC.data.len) |i|
    //         std.debug.print("\n gemm C[{d}] {d}", .{ i, CC.data[i] });
    // }

    try lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

    return result;
}

/// Lean version of gemm, output Tensor must be preconstructed and 0 filled
/// NOTE: (IMPORTANT FOR CODE GEN) Since multibatch/multichannel is not supported by mat_mul neither gemm does. Remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
pub fn lean_gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: T, beta: T, transA: bool, transB: bool, result: *Tensor(T)) !void {
    //std.debug.print("\n[DEBUG] lean_gemm:", .{});
    //std.debug.print("\n  A shape: ", .{});
    //for (A.shape) |s| std.debug.print("{d} ", .{s});

    //std.debug.print("\n  B shape: ", .{});
    //for (B.shape) |s| std.debug.print("{d} ", .{s});

    //std.debug.print("\n  Result shape: ", .{});
    //for (result.shape) |s| std.debug.print("{d} ", .{s});

    //std.debug.print("\n  Alpha: {d}, Beta: {d}", .{ alpha, beta });
    //std.debug.print("\n  TransA: {}, TransB: {}", .{ transA, transB });

    var actual_A: Tensor(T) = undefined;
    var actual_B: Tensor(T) = undefined;
    var actual_A_ptr = A;
    var actual_B_ptr = B;

    // applying transposition
    if (transA) {
        //std.debug.print("\n  Transposing A...", .{});
        actual_A = try TensMath.transposeLastTwo(T, A);
        actual_A_ptr = &actual_A;
        //std.debug.print("\n  A shape after transpose: ", .{});
        //for (actual_A_ptr.shape) |s| std.debug.print("{d} ", .{s});
    }
    if (transB) {
        //std.debug.print("\n  Transposing B...", .{});
        actual_B = try TensMath.transposeLastTwo(T, B);
        actual_B_ptr = &actual_B;
        //std.debug.print("\n  B shape after transpose: ", .{});
        //for (actual_B_ptr.shape) |s| std.debug.print("{d} ", .{s});
    }

    // result = alpha * A * B
    //std.debug.print("\n  Performing matrix multiplication...", .{});
    try op_mat_mul.lean_mat_mul(T, actual_A_ptr, actual_B_ptr, result);
    //std.debug.print("\n  Applying alpha scaling...", .{});
    for (0..result.size) |i| {
        result.data[i] *= alpha;
    }

    // result = result + beta * C
    if (C) |actual_C_ptr| {
        //std.debug.print("\n  C shape: ", .{});
        //for (actual_C_ptr.shape) |s| std.debug.print("{d} ", .{s});

        if (beta != 0) {
            //std.debug.print("\n  Adding beta * C...", .{});
            // no broadcast necessary
            if (result.size == actual_C_ptr.size) {
                //std.debug.print("\n  No broadcast needed", .{});
                for (0..result.size) |i| {
                    result.data[i] += actual_C_ptr.data[i] * beta;
                }
            }
            // broadcast from C to result
            else {
                //std.debug.print("\n  Broadcasting C...", .{});
                const res_rows = result.shape[result.shape.len - 2];
                const res_cols = result.shape[result.shape.len - 1];

                // Determine whether broadcast on rows and cols is needed or not
                const c_rows = if (actual_C_ptr.shape.len >= 2) actual_C_ptr.shape[actual_C_ptr.shape.len - 2] else 1;
                const c_cols = if (actual_C_ptr.shape.len >= 1) actual_C_ptr.shape[actual_C_ptr.shape.len - 1] else 1;

                //std.debug.print("\n  res_rows: {d}, res_cols: {d}, c_rows: {d}, c_cols: {d}", .{ res_rows, res_cols, c_rows, c_cols });

                for (0..actual_A_ptr.shape[0]) |b| {
                    for (0..actual_A_ptr.shape[1]) |c| {
                        for (0..res_rows) |i| {
                            for (0..res_cols) |j| {
                                // index used on C
                                const ci = if (c_rows == 1) 0 else i;
                                const cj = if (c_cols == 1) 0 else j;

                                // summing the product of C and beta
                                const result_index = (((b * result.shape[1]) + c) * result.shape[2] + i) * result.shape[3] + j;
                                const c_index = (((b * actual_C_ptr.shape[1]) + c) * actual_C_ptr.shape[2] + ci) * actual_C_ptr.shape[3] + cj;
                                result.data[result_index] += actual_C_ptr.data[c_index] * beta;
                            }
                        }
                    }
                }
            }
        }
    }

    // Clean up transposed tensors after all operations are complete
    if (transA) actual_A.deinit();
    if (transB) actual_B.deinit();

    //std.debug.print("\n[DEBUG] lean_gemm completed\n", .{});
}

pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    std.debug.print("\n[DEBUG] transposeLastTwo:", .{});
    std.debug.print("\n  Input tensor shape: ", .{});
    for (tensor.shape) |s| std.debug.print("{d} ", .{s});

    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        std.debug.print("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
        return TensorMathError.InputTensorsWrongShape;
    }

    var rows: usize = undefined;
    var cols: usize = undefined;
    var total: usize = undefined;
    var newShape: []usize = undefined;

    if (tensor.shape.len == 2) {
        rows = tensor.shape[0];
        cols = tensor.shape[1];
        total = rows * cols;
        newShape = try pkg_allocator.alloc(usize, 2);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = cols;
        newShape[1] = rows;
    } else { // 4D case
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        rows = tensor.shape[2];
        cols = tensor.shape[3];
        total = batch * channel * rows * cols;
        newShape = try pkg_allocator.alloc(usize, 4);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = batch;
        newShape[1] = channel;
        newShape[2] = cols;
        newShape[3] = rows;
    }

    std.debug.print("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    std.debug.print("\n  New shape: ", .{});
    for (newShape) |s| std.debug.print("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    std.debug.print("\n  Transposing data...", .{});

    if (tensor.shape.len == 2) {
        // Simple 2D transpose - Fixed indexing
        for (0..rows) |i| {
            for (0..cols) |j| {
                outData[j * rows + i] = tensor.data[i * cols + j];
            }
        }
    } else {
        // 4D transpose of last two dimensions
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        for (0..batch) |b| {
            for (0..channel) |c| {
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        const index_in = (((b * channel) + c) * rows + i) * cols + j;
                        const index_out = (((b * channel) + c) * cols + j) * rows + i;
                        outData[index_out] = tensor.data[index_in];
                    }
                }
            }
        }
    }

    std.debug.print("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
        .owns_memory = true,
    };
}
