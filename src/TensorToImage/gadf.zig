const std = @import("std");
const utils = @import("gaf_utils.zig");

pub const GadfError = error{
    InputTooShort,
};

/// LEAN VERSION — inline, zero dynamic allocations.
///
/// Computes the GADF matrix directly into a pre-allocated output buffer using
/// the algebraic form: G[i][j] = sqrt(1-x_i^2)*x_j - x_i*sqrt(1-x_j^2).
/// This is the expansion of sin(φ_i - φ_j), where φ = arccos(x_tilde).
/// Pre-calculates the sine components to achieve O(N) trigonometric overhead instead of O(N^2).
///
/// Normalization range affects angular coverage:
///   [-1, 1] → φ ∈ [0, π]   — full angular range, standard GADF
///   [ 0, 1] → φ ∈ [0, π/2] — half angular range, bijective but compressed
/// Both are mathematically valid; choose based on whether you need the bijective
/// inverse map ([0,1]) or maximum angular discriminability ([-1,1]).
///
/// Note on flat inputs: if all values in the series are equal, normalization
/// maps every point to x_tilde = 0, i.e. φ = π/2 for all i. The resulting
/// GADF matrix is all zeros (sin(φ_i - φ_j) = sin(0) = 0), which is correct —
/// a constant series has no angular difference. However, this is visually
/// indistinguishable from a non-flat series whose differences cancel symmetrically.
///
/// Parameters:
///   x_tilde      — normalized time series in [-1, 1] or [0, 1], length N
///   sines_buffer — pre-allocated temporary buffer of length N
///   gadf_out     — pre-allocated output buffer of length N * N (row-major)
pub fn lean_gadf(x_tilde: []const f32, sines_buffer: []f32, gadf_out: []f32) void {
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
    if (input.len < 2) return GadfError.InputTooShort;

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
    lean_gadf(x_tilde, sines_buffer, gadf_out);

    return gadf_out;
}
