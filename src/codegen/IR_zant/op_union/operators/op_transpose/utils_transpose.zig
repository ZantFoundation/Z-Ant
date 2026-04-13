const std = @import("std");
const IR_zant = @import("IR_zant");

const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Calculate the output shape for an ONNX transpose operation without performing the transpose
/// Handles implicit broadcasting if the permutation length is greater than the input rank.
pub fn get_transpose_output_shape(input_shape: []const usize, perm: ?[]const usize) ![]usize {
    const input_rank = input_shape.len;
    const perm_rank = if (perm) |p| p.len else input_rank; // Effective rank

    // Allocate space for the permutation array
    var real_perm = try pkg_allocator.alloc(usize, perm_rank);
    defer pkg_allocator.free(real_perm);

    // Validate perm if provided and build real_perm
    if (perm) |p| {
        if (p.len != perm_rank) {
            // This case should technically not happen if perm_rank is derived from p.len, but good practice
            std.log.warn("ERROR: perm length mismatch internal logic! p.len={} perm_rank={}\\n", .{ p.len, perm_rank });
            return error.InvalidPermutation;
        }

        // Validate that p is a valid permutation of [0..perm_rank)
        var used = try pkg_allocator.alloc(bool, perm_rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= perm_rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
        @memcpy(real_perm, p);
    } else {
        // If no perm given, reverse the dimension order for perm_rank
        for (0..perm_rank) |i| {
            real_perm[i] = perm_rank - 1 - i;
        }
    }

    // Create effective input shape (with leading 1s for broadcasting)
    var effective_input_shape = try pkg_allocator.alloc(usize, perm_rank);
    defer pkg_allocator.free(effective_input_shape);

    const leading_ones = if (perm_rank > input_rank) perm_rank - input_rank else 0;
    for (0..leading_ones) |i| {
        effective_input_shape[i] = 1;
    }
    @memcpy(effective_input_shape[leading_ones..], input_shape);

    // Allocate output shape array
    var output_shape = try pkg_allocator.alloc(usize, perm_rank);
    errdefer pkg_allocator.free(output_shape);

    // Fill output shape based on permuting the effective_input_shape
    for (0..perm_rank) |i| {
        const input_dim_index = real_perm[i];
        if (input_dim_index >= perm_rank) { // Sanity check
            std.log.warn("ERROR: real_perm[{}]={} is out of bounds for perm_rank={}\\n", .{ i, input_dim_index, perm_rank });
            return error.InvalidPermutation;
        }
        output_shape[i] = effective_input_shape[input_dim_index];
    }

    return output_shape;
}
