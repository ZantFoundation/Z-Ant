const std = @import("std");

/// Assigns each value in `input` to a quantile bin in [0, q-1].
/// Uses q-1 interior boundaries at sorted indices k*n/q (integer division) for k=1..q-1.
/// `sorted_buf`: caller-owned []f32 of length input.len (temp storage for sorted copy).
/// `bins_out`:   caller-owned []usize of length input.len.
pub fn quantileBins(
    input: []const f32,
    q: usize,
    sorted_buf: []f32,
    bins_out: []usize,
) void {
    const n = input.len;
    std.debug.assert(sorted_buf.len == n);
    std.debug.assert(bins_out.len == n);
    std.debug.assert(q > 0);

    @memcpy(sorted_buf, input);
    std.mem.sort(f32, sorted_buf, {}, std.sort.asc(f32));

    for (input, bins_out) |val, *bin| {
        var b: usize = 0;
        for (1..q) |k| {
            const edge_idx = (k * n) / q;
            if (val >= sorted_buf[edge_idx]) b += 1;
        }
        bin.* = @min(b, q - 1);
    }
}

/// Builds a row-stochastic Q×Q Markov transition matrix from bin assignments.
/// Counts single-step transitions: W[bins[t]][bins[t+1]] for t=0..n-2.
/// Row-normalizes; rows with zero sum are left as zero.
/// `matrix_out`: caller-owned []f32 of length q*q, row-major.
pub fn transitionMatrix(
    bins: []const usize,
    q: usize,
    matrix_out: []f32,
) void {
    std.debug.assert(matrix_out.len == q * q);

    @memset(matrix_out, 0.0);

    if (bins.len > 1) {
        for (0..bins.len - 1) |t| {
            matrix_out[bins[t] * q + bins[t + 1]] += 1.0;
        }
    }

    for (0..q) |r| {
        var sum: f32 = 0.0;
        for (0..q) |c| sum += matrix_out[r * q + c];
        if (sum > 0.0) {
            for (0..q) |c| matrix_out[r * q + c] /= sum;
        }
    }
}
