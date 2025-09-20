const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const quantScheme = zant.core.quantization.quantScheme;
const QuantTensMath = @import("quant_tensor_math_standard.zig");
const TensMath = @import("../TensorMath/tensor_math_standard.zig");

// Note that this function cuold benefit from SIMD optimizations

/// Implements the GEMM operator from the ONNX standard https://onnx.ai/onnx/operators/onnx__Gemm.html Y = alpha*A*B + beta*C
pub fn quant_gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: f32, beta: f32, transA: bool, transB: bool) !Tensor(T) {
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
    result.details = A.details; // Propagating details
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

    try quant_lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

    return result;
}

/// Lean version of gemm, output Tensor must be preconstructed and 0 filled
pub fn quant_lean_gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: T, beta: T, transA: bool, transB: bool, result: *Tensor(T)) !void {
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
    defer {
        if (transA) actual_A.deinit();
        if (transB) actual_B.deinit();
    }

    // result = A * B
    const vals_in_cache = std.atomic.cache_line / @sizeOf(T);
    if (B.shape[B.shape.len - 1] > vals_in_cache) {
        try QuantTensMath.quant_blocked_mat_mul_lean(T, actual_A_ptr, actual_B_ptr, result);
    } else {
        try QuantTensMath.quant_mat_mul_lean(T, actual_A_ptr, actual_B_ptr, result);
    }
    //std.debug.print("\n  Performing matrix multiplication...", .{});

    // result = alpha * result
    //std.debug.print("\n  Applying alpha scaling...", .{});
    switch (result.details) {
        .quant => |*qd| {
            // For quantized tensors: apply alpha scaling or clamp at min or max

            // METHOD 1: multiply and clamp
            // const max = std.math.maxInt(T);
            // const min = std.math.minInt(T);
            // for (0..result.size) |i| {
            //     const new_value = (result.data[i] - qd.zero_point) * alpha + qd.zero_point;
            //     result.data[i] = if (new_value > max) {
            //         max;
            //     } else if (new_value < min) {
            //         min;
            //     } else {
            //         new_value;
            //     };
            // }

            // METHOD 2: multiply the scale_factor
            qd.scale_factor *= alpha;
        },

        else => {
            return TensorError.NotQuantizedTensor;
        },
    }

    // result = result + beta * C
    if (C) |actual_C_ptr| {
        //std.debug.print("\n  C shape: ", .{});
        //for (actual_C_ptr.shape) |s| std.debug.print("{d} ", .{s});

        if (beta != 0) {

            // no broadcast necessary
            if (result.size == actual_C_ptr.size) {
                //std.debug.print("\n  No broadcast needed", .{});

                switch (actual_C_ptr.details) {
                    .quant => |*qd| {
                        // For quantized tensors: prepare C * beta and then pass to element-wise sum

                        // METHOD 1: multiply and clamp
                        // const max = std.math.maxInt(T);
                        // const min = std.math.minInt(T);
                        // for (0..actual_C_ptr.size) |i| {
                        //     const new_value = (actual_C_ptr.data[i] - qd.zero_point) * beta + qd.zero_point;
                        //     actual_C_ptr.data[i] = if (new_value > max) {
                        //         max;
                        //     } else if (new_value < min) {
                        //         min;
                        //     } else {
                        //         new_value;
                        //     };
                        // }

                        // METHOD 2: multiply the scale_factor
                        qd.scale_factor *= beta;
                    },
                }
                try QuantTensMath.quant_sum_tensors_lean(T, T, result, actual_C_ptr, result);

            }
            // broadcast from C to result
            else {
                const res_rows = result.shape[result.shape.len - 2];
                const res_cols = result.shape[result.shape.len - 1];

                // Determine whether broadcast on rows and cols is needed or not
                const c_rows = if (actual_C_ptr.shape.len >= 2) actual_C_ptr.shape[actual_C_ptr.shape.len - 2] else 1;
                const c_cols = if (actual_C_ptr.shape.len >= 1) actual_C_ptr.shape[actual_C_ptr.shape.len - 1] else 1;

                //std.debug.print("\n  res_rows: {d}, res_cols: {d}, c_rows: {d}, c_cols: {d}", .{ res_rows, res_cols, c_rows, c_cols });

                // Dequantize C and result
                const dequant_C = try Tensor(f32).fromShape(&pkg_allocator, actual_C_ptr.shape);
                defer dequant_C.deinit();
                QuantTensMath.lean_dequantize(T, f32, actual_C_ptr, dequant_C);

                const dequant_result = try Tensor(f32).fromShape(&pkg_allocator, result.shape);
                defer dequant_result.deinit();
                QuantTensMath.lean_dequantize(T, f32, result, dequant_result);

                // Compute indexes and sum
                for (0..actual_A_ptr.shape[0]) |b| {
                    for (0..actual_A_ptr.shape[1]) |c| {
                        for (0..res_rows) |i| {
                            for (0..res_cols) |j| {
                                // indexes
                                const ci = if (c_rows == 1) 0 else i;
                                const cj = if (c_cols == 1) 0 else j;

                                const result_index = (((b * dequant_result.shape[1]) + c) * dequant_result.shape[2] + i) * dequant_result.shape[3] + j;
                                const c_index = (((b * dequant_C.shape[1]) + c) * dequant_C.shape[2] + ci) * dequant_C.shape[3] + cj;
                                
                                // summing the product of C and beta
                                dequant_result.data[result_index] += dequant_C.data[c_index] * beta;
                            }
                        }
                    }
                }

                // Requantize result
                result.deinit();
                try QuantTensMath.lean_quantize_minmax(T, dequant_result, result, quantScheme.ASYM);
            }
        }
    }

}