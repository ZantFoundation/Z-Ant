const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

const CACHE_BLOCK_SIZE_BYTES: usize = std.atomic.cache_line;

pub inline fn blocked_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.log.debug("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.log.debug("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return error.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        // For 1D vectors, we treat it as a dot product
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return error.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var Y = try Tensor(T).fromShape(&allocator, out_shape);
        errdefer Y.deinit();

        @memset(Y.data, 0);

        // Since this is just a dot product, we'll calculate it directly
        var sum: T = 0;
        for (0..K) |k| {
            sum += A.data[k] * B.data[k];
        }

        Y.data[0] = sum;
        return Y;
    }

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.log.debug("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return error.InputTensorsWrongShape;
    }

    // The input tensors must have at least 2 dimensions
    if (dim_num < 2) {
        // std.log.debug("Error: Input tensors must have at least 2 dimensions. Got: {}\n", .{dim_num});
        return error.InputTensorsWrongShape;
    }

    // Create output tensor

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.log.debug("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return error.InputTensorsWrongShape;
    }

    // std.log.debug("Validation passed, proceeding with multiplication\n", .{});

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
    errdefer Y.deinit();

    // std.log.debug("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\n", .{});

    @memset(Y.data, 0);

    try blocked_mat_mul_lean(T, A, B, &Y);

    return Y;
}

//Loosely inspired from https://coffeebeforearch.github.io/2020/06/23/mmul.html
//Easy to implement, works, loses some efficiency on non-square matrices or really large B matrices
pub inline fn blocked_mat_mul_lean(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *const Tensor(T)) !void {
    const dim_num = A.shape.len;

    // Handle 1D tensors as special case
    if (dim_num == 1) {
        if (B.shape.len != 1) {
            return error.InputTensorDifferentShape;
        }

        // For 1D vectors, we treat them as a dot product
        // A is a 1D vector (1xK), B is a 1D vector (Kx1), C is a scalar (1x1)
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return error.InputTensorsWrongShape;
        }

        if (C.shape.len != 1) {
            return error.OutputTensorWrongShape;
        }
        if (C.shape[0] != 1) {
            return error.OutputTensorWrongShape;
        }

        var sum: T = 0;
        for (0..K) |k| {
            sum += A.data[k] * B.data[k];
        }

        C.data[0] = sum;
        return;
    }

    // Regular matrix multiplication for dim_num >= 2
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
    var a_vec: @Vector(VEC_WIDTH, T) = undefined;
    var b_vec: @Vector(VEC_WIDTH, T) = undefined;
    var c_vec: @Vector(VEC_WIDTH, T) = undefined;

    var c_chunk_column: usize = 0;

    while (c_chunk_column + cache_block_size <= nearest_c_cols) : (c_chunk_column += cache_block_size) {
        for (0..c_rows) |c_chunk_row| {
            var tile: usize = 0;
            while (tile < nearest_b_rows) : (tile += cache_block_size) {
                for (0..cache_block_size) |t_row| {
                    simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
                }
            }
            //Handle rows that are not a multiple of cache_block_size
            var last_tile: usize = 0;
            while (last_tile < remaining_b_rows) : (last_tile += 1) {
                simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
            }
        }
    }

    for (0..c_rows) |c_chunk_row| {
        var tile: usize = 0;
        while (tile < nearest_b_rows) : (tile += cache_block_size) {
            for (0..cache_block_size) |t_row| {
                simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
            }
        }

        //Handle rows that are not a multiple of cache_block_size
        var last_tile: usize = 0;
        while (last_tile < remaining_b_rows) : (last_tile += 1) {
            simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, nearest_b_rows, last_tile, c_chunk_column, c_chunk_row, &a_vec, &b_vec, &c_vec);
        }
    }
}

inline fn simd_tile_mul(
    comptime T: anytype,
    A_ptr: [*]T,
    a_cols: usize,
    B_ptr: [*]T,
    b_cols: usize,
    C_ptr: [*]T,
    c_cols: usize,
    tile: usize,
    t_row: usize,
    c_chunk_column: usize,
    c_chunk_row: usize,
    a_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
    b_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
    c_vec: *(@Vector(std.simd.suggestVectorLength(T) orelse 4, T)),
) void {
    const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);

    // Ensure that c_chunk_column + CACHE_BLOCK_SIZE does not exceed c_cols
    const end_col = @min(CACHE_BLOCK_SIZE, c_cols - c_chunk_column);

    const a_val = A_ptr[c_chunk_row * a_cols + tile + t_row];

    // Create a vector filled with the same value of A
    a_vec.* = @splat(a_val);

    // Iteration on columns in blocks of simd_lanes
    var t_col: usize = 0;
    while (t_col + VEC_WIDTH <= end_col) : (t_col += VEC_WIDTH) {

        // Load elements of B into a vector
        for (0..VEC_WIDTH) |i| {
            b_vec[i] = B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col + i];
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
        C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] +=
            A_ptr[c_chunk_row * a_cols + tile + t_row] *
            B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col];
    }
}
