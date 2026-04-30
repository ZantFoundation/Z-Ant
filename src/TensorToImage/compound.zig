const std = @import("std");
const lib = @import("mod.zig");



// Scratch buffers for zero-allocation processing

/// All intermediate buffers needed by lean_compound for one series of length n
/// with q MTF bins. Allocate once, reuse across many series.
pub const CompoundScratch = struct {
    norm_buf:    []f32,   // length n   — normalized series (shared by GASF and GADF)
    cosines_buf: []f32,   // length n   — √(1−xᵢ²) precomputed for GASF
    sines_buf:   []f32,   // length n   — √(1−xᵢ²) precomputed for GADF
    sorted_buf:  []f32,   // length n   — sorted input copy for MTF quantile edges
    bins_buf:    []usize, // length n   — quantile bin index per time step
    matrix_buf:  []f32,   // length q*q — row-stochastic MTF transition matrix

    pub fn init(allocator: std.mem.Allocator, n: usize, q: usize) !CompoundScratch {
        const norm_buf = try allocator.alloc(f32, n);
        errdefer allocator.free(norm_buf);
        const cosines_buf = try allocator.alloc(f32, n);
        errdefer allocator.free(cosines_buf);
        const sines_buf = try allocator.alloc(f32, n);
        errdefer allocator.free(sines_buf);
        const sorted_buf = try allocator.alloc(f32, n);
        errdefer allocator.free(sorted_buf);
        const bins_buf = try allocator.alloc(usize, n);
        errdefer allocator.free(bins_buf);
        const matrix_buf = try allocator.alloc(f32, q * q);
        return .{
            .norm_buf    = norm_buf,
            .cosines_buf = cosines_buf,
            .sines_buf   = sines_buf,
            .sorted_buf  = sorted_buf,
            .bins_buf    = bins_buf,
            .matrix_buf  = matrix_buf,
        };
    }

    pub fn deinit(self: CompoundScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.norm_buf);
        allocator.free(self.cosines_buf);
        allocator.free(self.sines_buf);
        allocator.free(self.sorted_buf);
        allocator.free(self.bins_buf);
        allocator.free(self.matrix_buf);
    }
};


// lean_compound — zero-allocation single-series core

/// Zero-allocation core. Computes GASF, GADF and MTF for one time series and
/// writes a CHW f32 tensor of shape [3, n, n] directly into `output`.
///
/// output layout (length 3*n*n):
///   output[0*n*n .. 1*n*n]  →  GASF  shifted to [0, 1] via (x+1)/2
///   output[1*n*n .. 2*n*n]  →  GADF  shifted to [0, 1] via (x+1)/2
///   output[2*n*n .. 3*n*n]  →  MTF   already in [0, 1], copied as-is
///
/// Caller is responsible for:
///   - allocating output (length 3*n*n) and pointing it at the right offset
///     inside a larger batch buffer when processing multiple series
///   - allocating scratch via CompoundScratch.init() and keeping it alive
///     across calls (scratch is fully overwritten every call, never read)
pub fn lean_compound(
    input: []const f32,
    norm: lib.gaf_utils.NormRange,
    q: usize,
    scratch: *CompoundScratch,
    output: []f32,
) void {
    const n = input.len;
    std.debug.assert(n > 0);
    std.debug.assert(output.len == 3 * n * n);
    std.debug.assert(scratch.norm_buf.len == n);
    std.debug.assert(scratch.matrix_buf.len == q * q);

    const pixels = n * n;

    // Normalize input once; both GASF and GADF consume the same normalized series.
    lib.gaf_utils.normalize(input, scratch.norm_buf, norm) catch unreachable;

    // GASF → channel 0, then shift [-1,1] → [0,1] in-place
    lib.gasf.lean_gasf(scratch.norm_buf, scratch.cosines_buf, output[0 * pixels .. 1 * pixels]);
    for (output[0 * pixels .. 1 * pixels]) |*v| v.* = (v.* + 1.0) * 0.5;

    // GADF → channel 1, then shift [-1,1] → [0,1] in-place
    lib.gadf.lean_gadf(scratch.norm_buf, scratch.sines_buf, output[1 * pixels .. 2 * pixels]);
    for (output[1 * pixels .. 2 * pixels]) |*v| v.* = (v.* + 1.0) * 0.5;

    // MTF → channel 2 (values are already in [0,1])
    lib.mtf.mtf.lean_mtf(input, q, scratch.sorted_buf, scratch.bins_buf, scratch.matrix_buf, output[2 * pixels .. 3 * pixels]);
}

// toRGBImageF32 — allocating single-series wrapper


/// Computes GASF, GADF and MTF from a raw time series and packs them into a
/// single contiguous CHW f32 tensor of shape [3, n, n] (length 3*n*n):
///
///   output[0*n*n .. 1*n*n]  →  GASF  (channel 0)
///   output[1*n*n .. 2*n*n]  →  GADF  (channel 1)
///   output[2*n*n .. 3*n*n]  →  MTF   (channel 2)
///
/// All channels are in [0.0, 1.0] — the standard input range for CNNs.
/// GASF and GADF are natively in [-1, 1] and are shifted via (x+1)/2.
/// MTF is natively in [0, 1] and is copied as-is.
///
/// Caller owns the returned slice (length 3*n*n) and must free it.
pub fn toRGBImageF32(
    allocator: std.mem.Allocator,
    input: []const f32,
    norm: lib.gaf_utils.NormRange,
    q: usize,
) ![]f32 {
    const n = input.len;
    const pixels = n * n;

    var scratch = try CompoundScratch.init(allocator, n, q);
    defer scratch.deinit(allocator);

    const out = try allocator.alloc(f32, 3 * pixels);
    lean_compound(input, norm, q, &scratch, out);
    return out;
}


// batchToRGBImageF32 — batch processing, NCHW layout

/// Processes B time series and returns a flat f32 buffer in NCHW layout:
///   shape [B, 3, N, N], length B * 3 * N * N
///
/// To access series b, channel c, row r, col col:
///   data[((b * 3 + c) * N + r) * N + col]
///
/// All series must have the same length N.
/// Scratch buffers are allocated once and reused across all B series — zero
/// heap activity inside the processing loop.
///
/// Caller owns the returned slice and must free it.
/// To use with Tensor(f32): Tensor(f32).fromArray(&alloc, result, &[_]usize{B, 3, N, N})
pub fn batchToRGBImageF32(
    allocator: std.mem.Allocator,
    inputs: []const []const f32,
    norm: lib.gaf_utils.NormRange,
    q: usize,
) ![]f32 {
    std.debug.assert(inputs.len > 0);
    const b = inputs.len;
    const n = inputs[0].len;
    for (inputs) |s| std.debug.assert(s.len == n);

    const chw = 3 * n * n;
    const out = try allocator.alloc(f32, b * chw);
    errdefer allocator.free(out);

    var scratch = try CompoundScratch.init(allocator, n, q);
    defer scratch.deinit(allocator);

    for (inputs, 0..) |series, i| {
        lean_compound(series, norm, q, &scratch, out[i * chw .. (i + 1) * chw]);
    }

    return out;
}
