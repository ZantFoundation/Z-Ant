const IR_zant = @import("IR_zant");
const std = @import("std");
const Tensor = IR_zant.core.tensor.Tensor;
const pkgAllocator = IR_zant.pkg_allocator;
const TensMath = IR_zant.core.math_standard;

const CACHE_BLOCK_SIZE_BYTES: usize = std.atomic.cache_line;

pub inline fn lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), Y: *Tensor(T)) !void {
    const DEFAULT_VECTOR_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    const dim_num = A.shape.len;

    const M = A.shape[dim_num - 2];
    const N = B.shape[dim_num - 1];
    const K = A.shape[dim_num - 1];

    // std.log.debug("\nMatrix multiplication dimensions: M={}, N={}, K={}\n", .{ M, N, K });
    // std.log.debug("Input tensor A shape: ", .{});
    // for (A.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\nInput tensor B shape: ", .{});
    // for (B.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\n", .{});

    // std.log.debug("Output tensor Y shape: ", .{});
    // for (Y.shape) |dim| std.log.debug("{} ", .{dim});
    // std.log.debug("\n", .{});

    // SIMD vector type
    const Vec = @Vector(DEFAULT_VECTOR_WIDTH, T);
    const VecOut = @Vector(DEFAULT_VECTOR_WIDTH, T);

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

                // Convert and multiply
                const a_vec: VecOut = @splat(@as(T, a_val));
                const b_vec_out: VecOut = @as(VecOut, b_vec);
                sum_vec += a_vec * b_vec_out;
            }

            // Store result
            comptime var v: usize = 0;
            inline while (v < DEFAULT_VECTOR_WIDTH) : (v += 1) {
                Y_ptr[out_idx + v] = sum_vec[v];
            }
        }

        // Handle remaining columns
        while (j < N) : (j += 1) {
            var sum: T = 0;
            const out_idx = out_offset + j;

            var k: usize = 0;
            while (k < K) : (k += 1) {
                sum += @as(T, A_ptr[row_offset + k]) *
                    @as(T, B_ptr[k * N + j]);
            }
            Y_ptr[out_idx] = sum;
        }
    }

    // std.log.debug("Matrix multiplication completed\n", .{});
}
