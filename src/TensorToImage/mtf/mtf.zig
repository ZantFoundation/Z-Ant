const std = @import("std");
const mtf_utils = @import("mtf_utils.zig");

pub const MtfError = error{
    InputTooShort,
    InvalidBins,
};

/// Zero-allocation core. All buffers are caller-owned and pre-allocated.
/// - sorted_buf: []f32 len=n  (temp: sorted copy of input for quantile edges)
/// - bins_buf:   []usize len=n (bin assignment per sample)
/// - matrix_buf: []f32 len=q*q (row-stochastic transition matrix, row-major)
/// - output:     []f32 len=n*n  (MTF result, row-major, values in [0,1])
pub fn lean_mtf(
    input: []const f32,
    q: usize,
    sorted_buf: []f32,
    bins_buf: []usize,
    matrix_buf: []f32,
    output: []f32,
) void {
    const n = input.len;
    std.debug.assert(sorted_buf.len == n);
    std.debug.assert(bins_buf.len == n);
    std.debug.assert(matrix_buf.len == q * q);
    std.debug.assert(output.len == n * n);

    mtf_utils.quantileBins(input, q, sorted_buf, bins_buf);
    mtf_utils.transitionMatrix(bins_buf, q, matrix_buf);

    for (0..n) |i| {
        for (0..n) |j| {
            output[i * n + j] = matrix_buf[bins_buf[i] * q + bins_buf[j]];
        }
    }
}

/// Standard wrapper. Allocates all intermediate buffers; caller owns returned slice.
/// Returns MtfError.InputTooShort if input.len < 2.
/// Returns MtfError.InvalidBins if q == 0.
pub fn mtf(
    allocator: std.mem.Allocator,
    input: []const f32,
    q: usize,
) ![]f32 {
    if (input.len < 2) return MtfError.InputTooShort;
    if (q == 0) return MtfError.InvalidBins;

    const n = input.len;

    const sorted_buf = try allocator.alloc(f32, n);
    defer allocator.free(sorted_buf);

    const bins_buf = try allocator.alloc(usize, n);
    defer allocator.free(bins_buf);

    const matrix_buf = try allocator.alloc(f32, q * q);
    defer allocator.free(matrix_buf);

    const output = try allocator.alloc(f32, n * n);
    errdefer allocator.free(output);

    lean_mtf(input, q, sorted_buf, bins_buf, matrix_buf, output);

    return output;
}
