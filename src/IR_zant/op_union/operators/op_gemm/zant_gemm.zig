const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const TensMath = @import("tensor_math_standard.zig");

// Note that this function cuold benefit from SIMD optimizations

/// Implements the GEMM operator from the ONNX standard https://onnx.ai/onnx/operators/onnx__Gemm.html Y = alpha*A*B + beta*C
/// NOTE: (IMPORTANT FOR CODE GEN) Since multibatch/multichannel is not supported by mat_mul neither gemm does. Remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
/// The broadcasting only accours in rows and cols dimensions, while batch and channel dimensions must have the same size between the operands.
pub fn gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: f32, beta: f32, transA: bool, transB: bool) !Tensor(T) {
    // Special case for 1D tensors (vectors)
    if (A.shape.len == 1 and B.shape.len == 1) {
        // Validate dimensions for vector dot product
        if ((transA and transB) or (!transA and !transB)) {
            // Both transposed or both not transposed: expect same length
            if (A.shape[0] != B.shape[0]) {
                return TensorMathError.InputTensorDimensionMismatch;
            }
        } else {
            // Do nothing special in this case
            // The vector dimensions will be checked in lean_mat_mul
        }

        // For 1D vectors, result is a scalar (1D tensor with size 1)
        var res_shape = try pkg_allocator.alloc(usize, 1);
        errdefer pkg_allocator.free(res_shape);
        res_shape[0] = 1;

        var result = try Tensor(T).fromShape(&pkg_allocator, res_shape);
        errdefer result.deinit();

        @memset(result.data, 0);
        try lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

        return result;
    }

    // Handle 2D tensors (matrices)
    if (A.shape.len == 2 and B.shape.len == 2) {
        var cond_A: usize = 0;
        var cond_B: usize = 0;
        var res_rows: usize = 0;
        var res_cols: usize = 0;

        // Get dimensions based on transposition flags
        if (transA) {
            cond_A = A.shape[0];
            res_rows = A.shape[1];
        } else {
            cond_A = A.shape[1];
            res_rows = A.shape[0];
        }

        if (transB) {
            cond_B = B.shape[1];
            res_cols = B.shape[0];
        } else {
            cond_B = B.shape[0];
            res_cols = B.shape[1];
        }

        if (cond_A != cond_B) {
            return TensorMathError.InputTensorDimensionMismatch;
        }

        // Check C tensor if provided
        if (C) |actual_C| {
            const c_shape_len = actual_C.shape.len;

            // C can be 2D or 1D (for broadcasting)
            if (c_shape_len > 2) {
                return TensorMathError.InputTensorsWrongShape;
            }

            // Validate C's dimensions for broadcasting
            if (c_shape_len == 2) {
                if ((actual_C.shape[0] != res_rows and actual_C.shape[0] != 1) or
                    (actual_C.shape[1] != res_cols and actual_C.shape[1] != 1))
                {
                    return TensorMathError.IncompatibleBroadcastShapes;
                }
            } else if (c_shape_len == 1) {
                // For 1D C, it must be a scalar (size 1) or match one full dimension
                if (actual_C.shape[0] != 1 and
                    actual_C.shape[0] != res_rows and
                    actual_C.shape[0] != res_cols)
                {
                    return TensorMathError.IncompatibleBroadcastShapes;
                }
            }
        }

        // Create result tensor
        var res_shape = try pkg_allocator.alloc(usize, 2);
        defer pkg_allocator.free(res_shape);
        res_shape[0] = res_rows;
        res_shape[1] = res_cols;

        var result = try Tensor(T).fromShape(&pkg_allocator, res_shape);
        errdefer result.deinit();

        @memset(result.data, 0);
        try lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

        return result;
    }

    // Original code for 4D tensors
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
    //     std.log.debug("\n gemm A[{d}] {d}", .{ i, A.data[i] });
    // for (0..B.data.len) |i|
    //     std.log.debug("\n gemm B[{d}] {d}", .{ i, B.data[i] });
    // if (C) |CC| {
    //     for (0..CC.data.len) |i|
    //         std.log.debug("\n gemm C[{d}] {d}", .{ i, CC.data[i] });
    // }

    try lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

    return result;
}

/// Lean version of gemm, output Tensor must be preconstructed and 0 filled
/// NOTE: (IMPORTANT FOR CODE GEN) Since multibatch/multichannel is not supported by mat_mul neither gemm does. Remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
pub fn lean_gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: T, beta: T, transA: bool, transB: bool, result: *Tensor(T)) !void {
    //std.log.debug("\n[DEBUG] lean_gemm:", .{});
    //std.log.debug("\n  A shape: ", .{});
    //for (A.shape) |s| std.log.debug("{d} ", .{s});

    //std.log.debug("\n  B shape: ", .{});
    //for (B.shape) |s| std.log.debug("{d} ", .{s});

    //std.log.debug("\n  Result shape: ", .{});
    //for (result.shape) |s| std.log.debug("{d} ", .{s});

    //std.log.debug("\n  Alpha: {d}, Beta: {d}", .{ alpha, beta });
    //std.log.debug("\n  TransA: {}, TransB: {}", .{ transA, transB });

    var actual_A: Tensor(T) = undefined;
    var actual_B: Tensor(T) = undefined;
    var actual_A_ptr = A;
    var actual_B_ptr = B;

    // applying transposition
    if (transA) {
        //std.log.debug("\n  Transposing A...", .{});
        actual_A = try TensMath.transposeLastTwo(T, A);
        actual_A_ptr = &actual_A;
        //std.log.debug("\n  A shape after transpose: ", .{});
        //for (actual_A_ptr.shape) |s| std.log.debug("{d} ", .{s});
    }
    if (transB) {
        //std.log.debug("\n  Transposing B...", .{});
        actual_B = try TensMath.transposeLastTwo(T, B);
        actual_B_ptr = &actual_B;
        //std.log.debug("\n  B shape after transpose: ", .{});
        //for (actual_B_ptr.shape) |s| std.log.debug("{d} ", .{s});
    }
    defer {
        if (transA) actual_A.deinit();
        if (transB) actual_B.deinit();
    }

    // Matrix multiplication (with SIMD optimization for larger matrices)
    const vals_in_cache = std.atomic.cache_line / @sizeOf(T);

    // Special case for 1D vector * 2D matrix
    if (actual_A_ptr.shape.len == 1 and actual_B_ptr.shape.len == 2) {
        // Handle vector-matrix multiplication manually
        // A is a vector [1, k], B is a matrix [k, n], result is a vector [1, n]
        const k = actual_A_ptr.shape[0];
        const n = actual_B_ptr.shape[1];

        if (result.shape.len != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }

        // Zero the result
        @memset(result.data, 0);

        // Compute result[j] = sum(A[i] * B[i,j]) for all i
        for (0..n) |j| {
            var sum: T = 0;
            for (0..k) |i| {
                sum += actual_A_ptr.data[i] * actual_B_ptr.data[i * n + j];
            }
            result.data[j] = sum;
        }
    } else if (actual_B_ptr.shape.len >= 2 and actual_B_ptr.shape[actual_B_ptr.shape.len - 1] > vals_in_cache) {
        try TensMath.blocked_mat_mul_lean(T, actual_A_ptr, actual_B_ptr, result);
    } else {
        try TensMath.mat_mul_lean(T, actual_A_ptr, actual_B_ptr, result);
    }

    // result = alpha * A * B
    //std.log.debug("\n  Performing matrix multiplication...", .{});

    //std.log.debug("\n  Applying alpha scaling...", .{});
    for (0..result.size) |i| {
        result.data[i] *= alpha;
    }

    // result = result + beta * C
    if (C) |actual_C_ptr| {
        //std.log.debug("\n  C shape: ", .{});
        //for (actual_C_ptr.shape) |s| std.log.debug("{d} ", .{s});

        if (beta != 0) {
            //std.log.debug("\n  Adding beta * C...", .{});
            // no broadcast necessary
            if (result.size == actual_C_ptr.size) {
                //std.log.debug("\n  No broadcast needed", .{});
                for (0..result.size) |i| {
                    result.data[i] += actual_C_ptr.data[i] * beta;
                }
            }
            // broadcast from C to result
            else {
                //std.log.debug("\n  Broadcasting C...", .{});

                // Handle 1D tensors
                if (result.shape.len == 1) {
                    // For 1D result, C must be scalar or 1D compatible
                    if (actual_C_ptr.shape.len == 1) {
                        // C is 1D
                        const c_size = actual_C_ptr.shape[0];
                        if (c_size == 1) {
                            // C is scalar, broadcast to whole result
                            for (0..result.size) |i| {
                                result.data[i] += actual_C_ptr.data[0] * beta;
                            }
                        } else if (c_size == result.shape[0]) {
                            // C has the same size as result, element-wise addition
                            for (0..result.size) |i| {
                                result.data[i] += actual_C_ptr.data[i] * beta;
                            }
                        } else {
                            return TensorMathError.IncompatibleBroadcastShapes;
                        }
                    } else {
                        return TensorMathError.IncompatibleBroadcastShapes;
                    }
                } else if (result.shape.len == 2) {
                    // 2D result tensor handling
                    const res_rows = result.shape[0];
                    const res_cols = result.shape[1];

                    // C can be 1D, 2D, or scalar
                    if (actual_C_ptr.shape.len == 1) {
                        // C is a 1D tensor
                        const c_size = actual_C_ptr.shape[0];

                        if (c_size == 1) {
                            // C is scalar, broadcast to entire result
                            const c_val = actual_C_ptr.data[0] * beta;
                            for (0..res_rows) |i| {
                                for (0..res_cols) |j| {
                                    result.data[i * res_cols + j] += c_val;
                                }
                            }
                        } else if (c_size == res_rows) {
                            // C is a column vector, broadcast across columns
                            for (0..res_rows) |i| {
                                const c_val = actual_C_ptr.data[i] * beta;
                                for (0..res_cols) |j| {
                                    result.data[i * res_cols + j] += c_val;
                                }
                            }
                        } else if (c_size == res_cols) {
                            // C is a row vector, broadcast across rows
                            for (0..res_rows) |i| {
                                for (0..res_cols) |j| {
                                    result.data[i * res_cols + j] += actual_C_ptr.data[j] * beta;
                                }
                            }
                        } else {
                            return TensorMathError.IncompatibleBroadcastShapes;
                        }
                    } else if (actual_C_ptr.shape.len == 2) {
                        // C is a 2D tensor - need to check if broadcasting is needed
                        const c_rows = actual_C_ptr.shape[0];
                        const c_cols = actual_C_ptr.shape[1];

                        // Check if the dimensions allow broadcasting
                        if ((c_rows != res_rows and c_rows != 1) or
                            (c_cols != res_cols and c_cols != 1))
                        {
                            return TensorMathError.IncompatibleBroadcastShapes;
                        }

                        // Perform broadcasting
                        for (0..res_rows) |i| {
                            const c_row = if (c_rows == 1) 0 else i;

                            for (0..res_cols) |j| {
                                const c_col = if (c_cols == 1) 0 else j;

                                result.data[i * res_cols + j] +=
                                    actual_C_ptr.data[c_row * c_cols + c_col] * beta;
                            }
                        }
                    } else {
                        return TensorMathError.InputTensorsWrongShape;
                    }
                } else {
                    // 3D and 4D tensors
                    const res_rows = result.shape[result.shape.len - 2];
                    const res_cols = result.shape[result.shape.len - 1];

                    // Determine whether broadcast on rows and cols is needed or not
                    const c_rows = if (actual_C_ptr.shape.len >= 2) actual_C_ptr.shape[actual_C_ptr.shape.len - 2] else 1;
                    const c_cols = if (actual_C_ptr.shape.len >= 1) actual_C_ptr.shape[actual_C_ptr.shape.len - 1] else 1;

                    //std.log.debug("\n  res_rows: {d}, res_cols: {d}, c_rows: {d}, c_cols: {d}", .{ res_rows, res_cols, c_rows, c_cols });

                    if (result.shape.len <= 4) {
                        // Set default values for batch and channel dimensions
                        const batch = if (result.shape.len >= 3) result.shape[0] else 1;
                        const channel = if (result.shape.len >= 4) result.shape[1] else 1;

                        for (0..batch) |b| {
                            for (0..channel) |c| {
                                for (0..res_rows) |i| {
                                    for (0..res_cols) |j| {
                                        // index used on C
                                        const ci = if (c_rows == 1) 0 else i;
                                        const cj = if (c_cols == 1) 0 else j;

                                        // Calculate indices based on tensor dimensions
                                        var result_index: usize = 0;
                                        var c_index: usize = 0;

                                        if (result.shape.len == 3) {
                                            result_index = (b * res_rows + i) * res_cols + j;
                                        } else { // 4D
                                            result_index = (((b * result.shape[1]) + c) * res_rows + i) * res_cols + j;
                                        }

                                        if (actual_C_ptr.shape.len == 1) {
                                            c_index = 0; // C is scalar or 1D array
                                        } else if (actual_C_ptr.shape.len == 2) {
                                            c_index = ci * actual_C_ptr.shape[1] + cj;
                                        } else if (actual_C_ptr.shape.len == 3) {
                                            c_index = (b * actual_C_ptr.shape[1] + ci) * actual_C_ptr.shape[2] + cj;
                                        } else if (actual_C_ptr.shape.len == 4) {
                                            c_index = (((b * actual_C_ptr.shape[1]) + c) * actual_C_ptr.shape[2] + ci) * actual_C_ptr.shape[3] + cj;
                                        }

                                        result.data[result_index] += actual_C_ptr.data[c_index] * beta;
                                    }
                                }
                            }
                        }
                    } else {
                        return TensorMathError.UnsupportedTensorDimensions;
                    }
                }
            }
        }
    }

    // Note: We already have a defer block above that handles cleanup
    //std.log.debug("\n[DEBUG] lean_gemm completed\n", .{});
}

pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    //std.log.debug("\n[DEBUG] transposeLastTwo:", .{});
    //std.log.debug("\n  Input tensor shape: ", .{});
    //for (tensor.shape) |s| std.log.debug("{d} ", .{s});

    // Special case for 1D tensors
    if (tensor.shape.len == 1) {
        // For 1D tensors, we just return a copy since transpose doesn't change anything
        var newShape = try pkg_allocator.alloc(usize, 1);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = tensor.shape[0];

        // Create a copy of the input data
        const outData = try pkg_allocator.alloc(T, tensor.size);
        errdefer pkg_allocator.free(outData);

        // Copy the data
        @memcpy(outData, tensor.data);

        return Tensor(T){
            .data = outData,
            .size = tensor.size,
            .shape = newShape,
            .allocator = &pkg_allocator,
        };
    }

    // Verifying correct shape for 2D and 4D tensors
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        //std.log.debug("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
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

    //std.log.debug("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    //std.log.debug("\n  New shape: ", .{});
    //for (newShape) |s| std.log.debug("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    //std.log.debug("\n  Transposing data...", .{});

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

    //std.log.debug("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
    };
}
