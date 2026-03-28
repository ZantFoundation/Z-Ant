const std = @import("std");
const gasf_utils = @import("gaf_utils.zig");

/// Errori possibili nella versione standard.
pub const GasfError = error{
    InputTooShort,
    OutputSizeMismatch,
};

/// LEAN VERSION — inline, zero dynamic allocations.
///
/// Computes the GASF matrix directly into a pre-allocated output buffer.
///
/// Parameters:
///   input  — normalized time series in [-1, 1], length n
///   output — pre-allocated buffer of size n*n (row-major, grayscale u8)
///
/// Algebraic form (avoids trig): G[i][j] = xᵢ·xⱼ - √(1-xᵢ²)·√(1-xⱼ²)
/// Pixel rescaling:              p = (G[i][j] + 1.0) / 2.0 * 255.0
///
/// Precondition: output.len == input.len * input.len
pub inline fn lean_gasf(input: []const f32, output: []u8) void {
    const n = input.len;
    std.debug.assert(output.len == n * n);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const xi = input[i];
        // Clamp for numerical safety before sqrt
        const xi_sq = @max(0.0, 1.0 - xi * xi);
        const sq_xi = @sqrt(xi_sq);

        var j: usize = 0;
        while (j < n) : (j += 1) {
            const xj = input[j];
            const xj_sq = @max(0.0, 1.0 - xj * xj);
            const sq_xj = @sqrt(xj_sq);

            // G[i][j] ∈ [-1, 1]
            const g_val = xi * xj - sq_xi * sq_xj;

            // Rescale → [0, 255]
            const pixel_f = (g_val + 1.0) / 2.0 * 255.0;
            output[i * n + j] = @intFromFloat(@round(pixel_f));
        }
    }
}

/// STANDARD VERSION — allocates internally, validates input, calls lean_gasf.
///
/// Parameters:
///   allocator — allocator used for the output and the temporary buffer
///   input     — raw time series (any value range)
///
/// Returns: []u8 slice of length n*n (raw grayscale, row-major).
///          The caller owns the returned slice and must free it.
///
/// Errors: GasfError.InputTooShort if input.len < 2
pub fn gasf(allocator: std.mem.Allocator, input: []const f32) ![]u8 {
    if (input.len < 2) return GasfError.InputTooShort;

    const n = input.len;

    // Temporary buffer for the normalized series
    const normalized = try allocator.alloc(f32, n);
    defer allocator.free(normalized);

    // Normalize to [-1, 1]
    gasf_utils.normalize_minmax(input, normalized);

    // Allocate n*n output
    const output = try allocator.alloc(u8, n * n);
    errdefer allocator.free(output);

    // Compute GASF
    lean_gasf(normalized, output);

    return output;
}
