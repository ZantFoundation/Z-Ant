const std = @import("std");
const zant = @import("../../../../zant.zig");
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;

// Calculate the output shape for element-wise multiplication with broadcasting.
// Allocates and returns the new shape. Caller owns the memory.
pub fn get_mul_output_shape(lhs: []const usize, rhs: []const usize) ![]usize {
    const rank1 = lhs.len;
    const rank2 = rhs.len;
    const max_rank = @max(rank1, rank2);

    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    // Pad shapes with 1s for broadcasting check
    var i: usize = 0;
    while (i < max_rank) : (i += 1) {
        const dim1 = if (rank1 + i >= max_rank) lhs[rank1 + i - max_rank] else 1;
        const dim2 = if (rank2 + i >= max_rank) rhs[rank2 + i - max_rank] else 1;

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            std.log.warn("Incompatible broadcast shapes: dim1={}, dim2={}, index={}\n", .{ dim1, dim2, i });
            std.log.warn("lhs shape: {any}\n", .{lhs});
            std.log.warn("rhs shape: {any}\n", .{rhs});
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[i] = @max(dim1, dim2);
    }

    return out_shape; // Return the newly allocated shape
}
