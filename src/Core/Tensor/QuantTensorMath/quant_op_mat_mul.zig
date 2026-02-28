const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const tensorType = zant.core.tensor.TensorType;
const tensorDetails = zant.core.tensor.TensorDetails;
const quantDetails = zant.core.tensor.QuantDetails;
const pkg_allocator = zant.utils.allocator.allocator;
const assert = std.debug.assert;

const ArchitectureError = zant.utils.error_handler.ArchitectureError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const error_handler = zant.utils.error_handler;
const TensorError = error_handler.TensorError;

// Optimize for L1 cache size (typically 32KB)
const BLOCK_SIZE_M: usize = 32;
const BLOCK_SIZE_N: usize = 32;
const BLOCK_SIZE_K: usize = 32;

// Use largest available SIMD width
// const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const UNROLL_FACTOR: usize = 10;

// TODO: add support for matrix multiplication for matrix distribuited in multi-batch/multi-channel tensors (for example of shape {2, 3, 5, 5}), now supports only tensors with shape {1, 1, N, M}
/// Performs classic matrix multiplication on given tensors using the least 2 dimensions
pub inline fn quant_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.debug.print("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.debug.print("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.debug.print("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    // The input tensors must have at least 2 dimensions
    if (dim_num < 2) {
        // std.debug.print("Error: Input tensors must have at least 2 dimensions. Got: {}\n", .{dim_num});
        return TensorMathError.InputTensorsWrongShape;
    }

    // Create output tensor

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.debug.print("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return TensorMathError.InputTensorsWrongShape;
    }

    // std.debug.print("Validation passed, proceeding with multiplication\n", .{});

    // Setup output tensor shape

    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    Y.details = A.details; // Propagating details
    errdefer Y.deinit();

    // std.debug.print("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\n", .{});

    @memset(Y.data, 0);

    try quant_lean_mat_mul(T, A, B, &Y);

    return Y;
}

pub inline fn quant_lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), Y: *Tensor(T)) !void {
    const DEFAULT_VECTOR_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    const dim_num = A.shape.len;

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Add dimension validation
    if (M >= std.math.maxInt(usize) / 2 or
        N >= std.math.maxInt(usize) / 2 or
        K >= std.math.maxInt(usize) / 2)
    {
        return TensorMathError.InputTensorsWrongShape;
    }

    // Add shape validation
    if (B.shape[dim_num - 2] != K) {
        return TensorMathError.InputTensorsWrongShape;
    }

    // Validate output tensor shape
    if (Y.shape[dim_num - 2] != M or Y.shape[dim_num - 1] != N) {
        return TensorMathError.OutputTensorWrongShape;
    }

    // Debug prints
    // std.debug.print("\nMatrix multiplication dimensions: M={}, N={}, K={}\n", .{ M, N, K });
    // std.debug.print("Input tensor A shape: ", .{});
    // for (A.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\nInput tensor B shape: ", .{});
    // for (B.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\nOutput tensor Y shape: ", .{});
    // for (Y.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\n", .{});

    // Get pointers for faster access
    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const Y_ptr = Y.data.ptr;

    // Main matrix multiplication loop with SIMD
    // Switch for tensor type

    switch (Y.details) {
        // Quantized tensor: int8 -> int32 -> int8 pipeline
        .quant => |*qd| {
            if (@typeInfo(T) == .int) {

                // SIMD vector type
                // The vector used for the product is of type T
                const Vec = @Vector(DEFAULT_VECTOR_WIDTH, T);
                // The vector used for accumulation is of type i32
                const VecOut = @Vector(DEFAULT_VECTOR_WIDTH, i32);

                // Max and min values for clamping
                //std.debug.print("\n\n\n{}\n\n{}\n\n\n", .{T, Y.details});
                const max_value = std.math.maxInt(T);
                const min_value = std.math.minInt(T);

                // For mat mul the output scale factor is the product of the input tensors scale factors
                qd.scale_factor = A.details.quant.scale_factor * B.details.quant.scale_factor;
                // zero_point
                qd.zero_point = 0; // This is a bit brutal but for better values we would need to analyze float execution of the model
                // effective_scale
                var effective_scale: f32 = A.details.quant.scale_factor * B.details.quant.scale_factor / qd.scale_factor;
                // shift_correction and effective_scale normalization in [1, 0.5] range
                var shift_correction: i6 = 0;
                while (effective_scale >= 1) {
                    effective_scale /= 2;
                    shift_correction -= 1;
                }
                while (effective_scale < 0.5) {
                    effective_scale *= 2;
                    shift_correction += 1;
                }
                // multiplier
                const shift: u6 = 31;

                // debug prints
                // std.debug.print("\neffective_scale = {}\n", .{effective_scale});
                // std.debug.print("shift = {}\n", .{shift});
                // std.debug.print("shift = {}\n", .{shift_correction});
                // std.debug.print("multiplier = {}\n", .{effective_scale * @as(f32, 1 << shift)});

                const multiplier: i32 = @intFromFloat(@round(effective_scale * @as(f32, 1 << shift)));

                // The actual multiplication with zero point adjustment
                var i: usize = 0;
                while (i < M) : (i += 1) {
                    // if (i % 100 == 0) std.debug.print("Processing row {}/{}\n", .{ i, M });
                    const row_offset = i * K;
                    const out_offset = i * N;

                    var j: usize = 0;
                    while (j + DEFAULT_VECTOR_WIDTH <= N) : (j += DEFAULT_VECTOR_WIDTH) {
                        var sum_vec: VecOut = @splat(0);
                        const out_idx = out_offset + j;

                        // Inner product with SIMD
                        var k: usize = 0;
                        while (k < K) : (k += 1) {
                            const a_val = A_ptr[row_offset + k] - A.details.quant.zero_point;
                            const b_offset = k * N + j;

                            // Load B values directly into vector
                            var b_vec: Vec = undefined;
                            comptime var v: usize = 0;
                            inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                                b_vec[v] = B_ptr[b_offset + v] - @as(T, @intCast(B.details.quant.zero_point));
                            }

                            // Convert and multiply
                            const a_vec: VecOut = @splat(@as(T, @intCast(a_val)));
                            const b_vec_out: VecOut = @as(VecOut, b_vec);
                            sum_vec += a_vec * b_vec_out;
                        }

                        // Store result
                        comptime var v: usize = 0;
                        inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                            // Apply multiplier and shift, add output_zero_point and then clamp
                            // const result = qd.zero_point + ((sum_vec[v] * multiplier) >> (shift + shift_correction));
                            const result = @as(i32, @intCast(qd.zero_point)) + @as(i32, @intCast((@as(i64, sum_vec[v]) * @as(i64, multiplier)) >> (@as(u6, @intCast(@as(i8, shift) + @as(i8, shift_correction))))));
                            Y_ptr[out_idx + v] = @as(T, @intCast(std.math.clamp(result, min_value, max_value)));
                        }
                    }

                    // Handle remaining columns
                    while (j < N) : (j += 1) {
                        var sum: i32 = 0;
                        const out_idx = out_offset + j;

                        var k: usize = 0;
                        while (k < K) : (k += 1) {
                            sum += @as(i32, @intCast(A_ptr[row_offset + k] - A.details.quant.zero_point)) *
                                @as(i32, @intCast(B_ptr[k * N + j] - B.details.quant.zero_point));
                        }

                        // std.debug.print("\nsum = {}\n", .{sum});
                        // std.debug.print("multiplier = {}\n", .{multiplier});
                        // std.debug.print("shift = {}\n", .{shift});
                        // std.debug.print("shift_correction = {}\n", .{shift_correction});
                        // std.debug.print("qd.zero_point = {}\n", .{qd.zero_point});
                        // std.debug.print("max_value = {}\n", .{max_value});
                        // std.debug.print("min_value = {}\n", .{min_value});
                        // std.debug.print("type = {}\n", .{T});

                        sum = @as(i32, @intCast(qd.zero_point)) + @as(i32, @intCast((@as(i64, sum) * @as(i64, multiplier)) >> (@as(u6, @intCast(@as(i8, shift) + @as(i8, shift_correction))))));
                        //std.debug.print("\nsum = {}\n", .{sum});

                        Y_ptr[out_idx] = @as(T, @intCast(std.math.clamp(sum, min_value, max_value)));
                    }
                }
            } else return TensorError.NotQuantizedTensor;
        },
        else => {
            return TensorError.NotQuantizedTensor;
        },
    }

    // std.debug.print("Matrix multiplication completed\n", .{});
}

const CACHE_BLOCK_SIZE_BYTES: usize = std.atomic.cache_line;

pub inline fn quant_blocked_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.debug.print("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.debug.print("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.debug.print("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    // The input tensors must have at least 2 dimensions
    if (dim_num < 2) {
        // std.debug.print("Error: Input tensors must have at least 2 dimensions. Got: {}\n", .{dim_num});
        return TensorMathError.InputTensorsWrongShape;
    }

    // Create output tensor

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.debug.print("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return TensorMathError.InputTensorsWrongShape;
    }

    // std.debug.print("Validation passed, proceeding with multiplication\n", .{});

    // Setup output tensor shape

    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);
    errdefer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    Y.details = A.details; // Propagating details
    errdefer Y.deinit();

    // std.debug.print("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\n", .{});

    @memset(Y.data, 0);

    try quant_lean_blocked_mat_mul(T, A, B, &Y);

    return Y;
}

//Loosely inspired from https://coffeebeforearch.github.io/2020/06/23/mmul.html
//Easy to implement, works, loses some efficiency on non-square matrices or really large B matrices
pub inline fn quant_lean_blocked_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *Tensor(T)) !void {
    const cache_block_size = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const a_rows = A.shape[A.shape.len - 2];
    const a_cols = A.shape[A.shape.len - 1];

    const b_cols = B.shape[B.shape.len - 1];
    const b_rows = a_cols;

    const c_rows = a_rows;
    const c_cols = b_cols;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;

    const nearest_c_cols = cache_block_size * (c_cols / cache_block_size);
    //const remaining_c_cols = c_cols - nearest_c_cols;
    const nearest_b_rows = cache_block_size * (b_rows / cache_block_size);
    const remaining_b_rows = b_rows - nearest_b_rows;

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);

    var zero: i32 = undefined;
    var scale: f32 = undefined;

    switch (C.details) {

        // Quantized tensor: int8 -> int32 -> int8 pipeline
        .quant => {
            if (@typeInfo(T) == .int) {

                // The vector used for the product is of type T
                var a_vec: @Vector(VEC_WIDTH, T) = undefined;
                var b_vec: @Vector(VEC_WIDTH, T) = undefined;
                // The vector used for accumulation is of type i32
                var c_vec: @Vector(VEC_WIDTH, i32) = undefined;

                var c_chunk_column: usize = 0;

                // Max and min values for clamping
                const max_value = std.math.maxInt(T);
                const min_value = std.math.minInt(T);

                // For mat mul the output scale factor is the product of the input tensors scale factors
                scale = A.details.quant.scale_factor * B.details.quant.scale_factor;
                // effective_scale
                var effective_scale: f32 = A.details.quant.scale_factor * B.details.quant.scale_factor / scale;
                // shift_correction and effective_scale normalization in [1, 0.5] range
                var shift_correction: i6 = 0;
                while (effective_scale >= 1) {
                    effective_scale /= 2;
                    shift_correction -= 1;
                }
                while (effective_scale < 0.5) {
                    effective_scale *= 2;
                    shift_correction += 1;
                }
                // multiplier
                const shift: u6 = 31;
                const multiplier: i32 = @intFromFloat(@round(effective_scale * @as(f32, 1 << shift)));

                // zero_point
                zero = 0; // This is a bit brutal but for better values we would need to analyze float execution of the model
                // accumulator needed because of how blocked_mat_mul works
                const c_accumulator = try pkg_allocator.alloc(i32, C.size);
                @memset(c_accumulator, 0);

                defer pkg_allocator.free(c_accumulator);

                while (c_chunk_column + cache_block_size <= nearest_c_cols) : (c_chunk_column += cache_block_size) {
                    for (0..c_rows) |c_chunk_row| {
                        var tile: usize = 0;
                        while (tile < nearest_b_rows) : (tile += cache_block_size) {
                            for (0..cache_block_size) |t_row| {
                                quant_simd_tile_mul(T, i32, A_ptr, a_cols, B_ptr, b_cols, c_accumulator.ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec, A.details.quant.zero_point, B.details.quant.zero_point);
                            }
                        }
                        //Handle rows that are not a multiple of cache_block_size
                        var last_tile: usize = 0;
                        while (last_tile < remaining_b_rows) : (last_tile += 1) {
                            quant_simd_tile_mul(T, i32, A_ptr, a_cols, B_ptr, b_cols, c_accumulator.ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec, A.details.quant.zero_point, B.details.quant.zero_point);
                        }
                    }
                }

                for (0..c_rows) |c_chunk_row| {
                    var tile: usize = 0;
                    while (tile < nearest_b_rows) : (tile += cache_block_size) {
                        for (0..cache_block_size) |t_row| {
                            quant_simd_tile_mul(T, i32, A_ptr, a_cols, B_ptr, b_cols, c_accumulator.ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec, A.details.quant.zero_point, B.details.quant.zero_point);
                        }
                    }

                    //Handle rows that are not a multiple of cache_block_size
                    var last_tile: usize = 0;
                    while (last_tile < remaining_b_rows) : (last_tile += 1) {
                        quant_simd_tile_mul(T, i32, A_ptr, a_cols, B_ptr, b_cols, c_accumulator.ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec, A.details.quant.zero_point, B.details.quant.zero_point);
                    }
                }

                // After accumulating
                for (0..C.size) |i| {
                    // Apply multiplier and shift, add output_zero_point and then clamp
                    const result = @as(i32, @intCast(zero)) + @as(i32, @intCast((@as(i64, c_accumulator[i]) * @as(i64, multiplier)) >> (@as(u6, @intCast(@as(i8, shift) + @as(i8, shift_correction))))));
                    C_ptr[i] = @as(T, @intCast(std.math.clamp(result, min_value, max_value)));
                }
            }
        },
        else => return TensorError.NotQuantizedTensor,
    }
    C.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = scale,
            .zero_point = zero,
        },
    };
}

inline fn quant_simd_tile_mul(comptime T: anytype, comptime T1: type, A_ptr: [*]T, a_cols: usize, B_ptr: [*]T, b_cols: usize, C_ptr: [*]T1, c_cols: usize, tile: usize, t_row: usize, c_chunk_column: usize, c_chunk_row: usize, a_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)), b_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)), c_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, i32)), a_zero_point: isize, b_zero_point: isize) void {
    const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    //const VEC_WIDTH: usize = comptime (8);

    // Ensure that c_chunk_column + CACHE_BLOCK_SIZE does not exceed c_cols
    const end_col = @min(CACHE_BLOCK_SIZE, c_cols - c_chunk_column);

    // temporary clamping
    const max_value = if (@typeInfo(T) == .int) std.math.maxInt(T) else 0;
    const min_value = if (@typeInfo(T) == .int) std.math.minInt(T) else 0;

    var a_val = A_ptr[c_chunk_row * a_cols + tile + t_row] - a_zero_point;
    if (a_val > max_value) {
        a_val = max_value;
    } else if (a_val < min_value) {
        a_val = min_value;
    }

    // Create a vector filled with the same value of A
    a_vec.* = @splat(@as(T, @intCast(a_val)));
    // var b_vec: @Vector(VEC_WIDTH, T) = undefined;
    // var c_vec: @Vector(VEC_WIDTH, T) = undefined;

    // Iteration on columns in blocks of simd_lanes
    var t_col: usize = 0;
    while (t_col + VEC_WIDTH <= end_col) : (t_col += VEC_WIDTH) {

        // Load elements of B into a vector
        for (0..VEC_WIDTH) |i| {
            const b_val = B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col + i] - b_zero_point;
            if (b_val > max_value) {
                b_vec[i] = max_value;
            } else if (b_val < min_value) {
                b_vec[i] = min_value;
            } else {
                b_vec[i] = @as(T, @intCast(b_val));
            }
        }

        // Load current values of C
        for (0..VEC_WIDTH) |i| {
            c_vec[i] = C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i];
        }

        // Multiply and accumulate
        c_vec.* += a_vec.* * b_vec.*;

        // Write the result in C
        for (0..VEC_WIDTH) |i| {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i] = c_vec[i];
        }
    }

    //Handle remaining columns without SIMD
    while (t_col < end_col) : (t_col += 1) {
        const c_val =
            (A_ptr[c_chunk_row * a_cols + tile + t_row] - a_zero_point) *
            (B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col] - b_zero_point);
        if (c_val > max_value) {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] += max_value;
        } else if (c_val < min_value) {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] += min_value;
        } else {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] += @as(i32, @intCast(c_val));
        }
    }
}

pub fn get_quant_mat_mul_output_shape(shape_a: []const usize, shape_b: []const usize) ![]usize {
    if (shape_a.len < 2 or shape_b.len < 2) {
        return error.InvalidShape;
    }

    const a_rows = shape_a[shape_a.len - 2];
    const a_cols = shape_a[shape_a.len - 1];
    const b_rows = shape_b[shape_b.len - 2];
    const b_cols = shape_b[shape_b.len - 1];

    if (a_cols != b_rows) {
        return error.ShapeMismatch;
    }

    var output_shape = try pkg_allocator.alloc(usize, shape_a.len);
    errdefer pkg_allocator.free(output_shape);

    // Copy batch dimensions from input_a
    for (0..shape_a.len - 2) |i| {
        output_shape[i] = shape_a[i];
    }

    // Set matrix multiplication dimensions
    output_shape[output_shape.len - 2] = a_rows;
    output_shape[output_shape.len - 1] = b_cols;

    return output_shape;
}
