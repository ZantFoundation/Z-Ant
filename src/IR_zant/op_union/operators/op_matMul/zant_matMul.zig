const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const assert = std.debug.assert;

const ArchitectureError = zant.utils.error_handler.ArchitectureError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const Uops = zant.uops;

const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

// Optimize for L1 cache size (typically 32KB)
const BLOCK_SIZE_M: usize = 32;
const BLOCK_SIZE_N: usize = 32;
const BLOCK_SIZE_K: usize = 32;

// Use largest available SIMD width
// const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const UNROLL_FACTOR: usize = 10;

// TODO: add support for matrix multiplication for matrix distribuited in multi-batch/multi-channel tensors (for example of shape {2, 3, 5, 5}), now supports only tensors with shape {1, 1, N, M}
/// Performs classic matrix multiplication on given tensors using the least 2 dimensions
pub inline fn mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T)) !Tensor(T) {
    // std.log.debug("\nStarting matrix multiplication validation...\n", .{});

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        // std.log.debug("Error: Input tensors have different dimensions. A: {}, B: {}\n", .{ A.shape.len, B.shape.len });
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // Special handling for 1D tensors (vectors)
    if (dim_num == 1) {
        // For 1D vectors, we treat it as a dot product
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        // Create a scalar output (1x1 tensor)
        const allocator = pkg_allocator;
        var out_shape = try allocator.alloc(usize, 1);
        defer allocator.free(out_shape);
        out_shape[0] = 1;

        var Y = try Tensor(T).fromShape(&allocator, out_shape);
        errdefer Y.deinit();

        @memset(Y.data, 0);
        try mat_mul_lean(T, A, B, &Y);

        return Y;
    }

    // For tensors with >= 2 dimensions

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        // std.log.debug("Error: Incompatible matrix dimensions for multiplication. A[{}]={}, B[{}]={}\n", .{ dim_num - 1, A.shape[dim_num - 1], dim_num - 2, B.shape[dim_num - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    // Create output tensor
    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // Check if the input tensors are empty
    if (M * N == 0 or K == 0) {
        // std.log.debug("Error: Empty input tensors. M={}, N={}, K={}\n", .{ M, N, K });
        return TensorMathError.InputTensorsWrongShape;
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

    try mat_mul_lean(T, A, B, &Y);

    return Y;
}

pub inline fn mat_mul_lean(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), Y: *Tensor(T)) !void {
    const DEFAULT_VECTOR_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    const dim_num = A.shape.len;

    // Handle 1D tensors as special case
    if (dim_num == 1) {
        if (B.shape.len != 1) {
            return TensorMathError.InputTensorDifferentShape;
        }

        // For 1D vectors, we treat them as a dot product
        // A is a 1D vector (1xK), B is a 1D vector (Kx1), Y is a scalar (1x1)
        const K = A.shape[0];

        if (K != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }

        if (Y.shape.len != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }
        if (Y.shape[0] != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }

        // Use wider type for computation to prevent overflow
        const ComputeType1D = switch (@typeInfo(T)) {
            .int => |int_info| switch (int_info.bits) {
                8 => if (int_info.signedness == .signed) i32 else u32,
                16 => if (int_info.signedness == .signed) i64 else u64,
                32 => if (int_info.signedness == .signed) i64 else u64,
                else => T,
            },
            .float => T,
            else => T,
        };

        var sum: ComputeType1D = 0;
        for (0..K) |k| {
            const a_val = @as(ComputeType1D, A.data[k]);
            const b_val = @as(ComputeType1D, B.data[k]);
            sum += a_val * b_val;
        }

        if (@typeInfo(T) == .int) {
            // Clamp to valid range for integer types
            const max_val = std.math.maxInt(T);
            const min_val = std.math.minInt(T);
            const clamped = std.math.clamp(sum, min_val, max_val);
            Y.data[0] = @as(T, @intCast(clamped));
        } else {
            Y.data[0] = @as(T, sum);
        }
        return;
    }

    // Regular matrix multiplication for dim_num >= 2
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

    // Debug prints only when needed
    if (false) {
        std.log.debug("\nMatrix multiplication dimensions: M={}, N={}, K={}\n", .{ M, N, K });
        std.log.debug("Input tensor A shape: ", .{});
        for (A.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\nInput tensor B shape: ", .{});
        for (B.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\nOutput tensor Y shape: ", .{});
        for (Y.shape) |dim| std.log.debug("{} ", .{dim});
        std.log.debug("\n", .{});
    }

    // SIMD vector type - use wider type for integer computations to prevent overflow
    const ComputeType = switch (@typeInfo(T)) {
        .int => |int_info| switch (int_info.bits) {
            8 => if (int_info.signedness == .signed) i32 else u32,
            16 => if (int_info.signedness == .signed) i64 else u64,
            32 => if (int_info.signedness == .signed) i64 else u64,
            else => T,
        },
        .float => T,
        else => T,
    };

    const Vec = @Vector(DEFAULT_VECTOR_WIDTH, T);
    const VecOut = @Vector(DEFAULT_VECTOR_WIDTH, ComputeType);

    // Get pointers for faster access
    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const Y_ptr = Y.data.ptr;

    // Main matrix multiplication loop with SIMD
    var i: usize = 0;
    while (i < M) : (i += 1) {
        // if (i % 100 == 0) std.log.debug("Processing row {}/{}\n", .{ i, M });
        const row_offset = i * K;
        const out_offset = i * N;

        var j: usize = 0;
        while (j + DEFAULT_VECTOR_WIDTH <= N) : (j += DEFAULT_VECTOR_WIDTH) {
            var sum_vec: VecOut = @splat(0);
            const out_idx = out_offset + j;

            // Inner product with SIMD
            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_val = A_ptr[row_offset + k];
                const b_offset = k * N + j;

                // Load B values directly into vector
                var b_vec: Vec = undefined;
                comptime var v: usize = 0;
                inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                    b_vec[v] = B_ptr[b_offset + v];
                }

                // Convert and multiply with overflow protection
                const a_vec: VecOut = @splat(@as(ComputeType, a_val));
                const b_vec_out: VecOut = blk: {
                    var result: VecOut = undefined;
                    comptime var idx: usize = 0;
                    inline while (idx < DEFAULT_VECTOR_WIDTH) : (idx += 1) {
                        result[idx] = @as(ComputeType, b_vec[idx]);
                    }
                    break :blk result;
                };
                sum_vec += a_vec * b_vec_out;
            }

            // Store result with type conversion
            comptime var v: usize = 0;
            inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                if (@typeInfo(T) == .int) {
                    // Clamp to valid range for integer types
                    const max_val = std.math.maxInt(T);
                    const min_val = std.math.minInt(T);
                    const clamped = std.math.clamp(sum_vec[v], min_val, max_val);
                    Y_ptr[out_idx + v] = @as(T, @intCast(clamped));
                } else {
                    Y_ptr[out_idx + v] = @as(T, sum_vec[v]);
                }
            }
        }

        // Handle remaining columns
        while (j < N) : (j += 1) {
            var sum: ComputeType = 0;
            const out_idx = out_offset + j;

            var k: usize = 0;
            while (k < K) : (k += 1) {
                const a_val = @as(ComputeType, A_ptr[row_offset + k]);
                const b_val = @as(ComputeType, B_ptr[k * N + j]);
                sum += a_val * b_val;
            }

            if (@typeInfo(T) == .int) {
                // Clamp to valid range for integer types
                const max_val = std.math.maxInt(T);
                const min_val = std.math.minInt(T);
                const clamped = std.math.clamp(sum, min_val, max_val);
                Y_ptr[out_idx] = @as(T, @intCast(clamped));
            } else {
                Y_ptr[out_idx] = @as(T, sum);
            }
        }
    }

    // std.log.debug("Matrix multiplication completed\n", .{});
}
