const std = @import("std");
const gasf_utils = @import("gaf_utils.zig");

/// Errori possibili nella versione standard.
pub const GasfError = error{
    InputTooShort,
    OutputSizeMismatch,
};

/// LEAN VERSION — zero dynamic allocations. All buffers are caller-owned.
///
/// Parameters:
///   input          — normalized time series in [-1, 1], length n
///   cosines_buffer — pre-allocated temp buffer of length n (√(1-xᵢ²) per sample)
///   output         — pre-allocated output buffer of size n*n (row-major, f32)
///
/// Algebraic form (avoids trig): G[i][j] = xᵢ·xⱼ - √(1-xᵢ²)·√(1-xⱼ²)
/// Pre-calculates cosine components in O(n) before the O(n²) matrix loop.
pub fn lean_gasf(input: []const f32, cosines_buffer: []f32, output: []f32) void {
    const n = input.len;
    std.debug.assert(n > 0);
    std.debug.assert(cosines_buffer.len == n);
    std.debug.assert(output.len == n * n);

    // Pre-compute √(1-xᵢ²) for all i in O(n)
    for (input, cosines_buffer) |x, *c| {
        c.* = @sqrt(@max(0.0, 1.0 - x * x));
    }

    // Fill the n×n matrix using pre-computed values
    for (0..n) |i| {
        for (0..n) |j| {
            output[i * n + j] = input[i] * input[j] - cosines_buffer[i] * cosines_buffer[j];
        }
    }
}

/// STANDARD VERSION — allocates internally, validates input, calls lean_gasf.
///
/// Parameters:
///   allocator — allocator used for the output and the temporary buffer
///   input     — raw time series (any value range)
///   norm      — gasf_utils.NormRange variant to pick the scale
///
/// Returns: []f32 slice of length n*n (row-major).
///          The caller owns the returned slice and must free it.
///
/// Errors: GasfError.InputTooShort if input.len < 2
pub fn gasf(allocator: std.mem.Allocator, input: []const f32, norm: gasf_utils.NormRange) ![]f32 {
    if (input.len < 2) return GasfError.InputTooShort;

    const n = input.len;

    // Temporary buffer for the normalized series
    const normalized = try allocator.alloc(f32, n);
    defer allocator.free(normalized);

    // Normalize the input data
    try gasf_utils.normalize(input, normalized, norm);

    // Temporary buffer for pre-computed cosine components √(1-xᵢ²)
    const cosines_buffer = try allocator.alloc(f32, n);
    defer allocator.free(cosines_buffer);

    // Allocate n*n output
    const output = try allocator.alloc(f32, n * n);
    errdefer allocator.free(output);

    // Compute GASF
    lean_gasf(normalized, cosines_buffer, output);

    return output;
}
