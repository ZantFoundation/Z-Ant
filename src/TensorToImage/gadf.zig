const std = @import("std");
const utils = @import("gaf_utils.zig");

/// LEAN VERSION — inline, zero dynamic allocations.
///
/// Computes the GADF matrix directly into a pre-allocated output buffer using
/// the algebraic form: G[i][j] = sqrt(1-x_i^2)*x_j - x_i*sqrt(1-x_j^2).
/// Pre-calculates the sine components to achieve O(N) trigonometric overhead instead of O(N^2).
///
/// Parameters:
///   x_tilde      — normalized time series in [-1, 1] or [0, 1], length N
///   sines_buffer — pre-allocated temporary buffer of length N
///   gadf_out     — pre-allocated output buffer of length N * N (row-major)
pub fn lean_gadf(x_tilde: []const f32, sines_buffer: []f32, gadf_out: []f32) !void {
    const n = x_tilde.len;

    //Debug assertion
    std.debug.assert(n > 0);
    std.debug.assert(sines_buffer.len == n);
    std.debug.assert(gadf_out.len == n * n);

    // Pre-compute sqrt(1 - x^2)
    for (x_tilde, sines_buffer) |x, *s| {
        s.* = @sqrt(@max(0.0, 1.0 - (x * x)));
    }

    // Compute the quasi-Gramian matrix using the formula sqrt(1-x_i^2)*x_j - x_i*sqrt(1-x_j^2)
    // quasi-Gramian due to non satisfying the linearity property (see https://arxiv.org/pdf/1506.00327.pdf, paragraph 2.1)
    for (0..n) |i| {
        for (0..n) |j| {
            const index = i * n + j;
            gadf_out[index] = (sines_buffer[i] * x_tilde[j]) - (x_tilde[i] * sines_buffer[j]);
        }
    }
}

/// STANDARD WRAPPER — Handles memory allocation and validation automatically.
///
/// Computes the Gramian Angular Difference Field (GADF) matrix for a given time series.
/// Normalizes the input to either [0, 1] or [-1, 1] before generating the N*N flattened matrix.
///
/// **Memory Warning:** The caller owns the returned `[]f32` slice and must explicitly
/// call `allocator.free()` on it to prevent memory leaks.
pub fn gadf(allocator: std.mem.Allocator, input: []const f32, norm: utils.NormRange) ![]f32 {
    const n = input.len;

    //Allocate temporary buffer for normalized data
    const x_tilde = try allocator.alloc(f32, n);
    defer allocator.free(x_tilde);

    //Normalize the input data
    try utils.normalize(input, x_tilde, norm);

    //Allocate temporary buffer for the sine pre-computation
    const sines_buffer = try allocator.alloc(f32, n);
    defer allocator.free(sines_buffer);

    //Allocate output matrix
    const gadf_out = try allocator.alloc(f32, n * n);
    errdefer allocator.free(gadf_out);

    //Execute the zero-allocation core math
    try lean_gadf(x_tilde, sines_buffer, gadf_out);

    return gadf_out;
}
