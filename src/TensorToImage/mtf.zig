const std = @import("std");
const quantile = @import("quantile.zig");

/// Main function to generate a Markov Transition Field from a time series.
/// 'n' is the time series length. 'Q' is the number of quantile bins.
/// Returns a flat []u8 matrix of size n*n representing grayscale pixels.
pub fn mtf(
    comptime n: usize,
    comptime Q: usize,
    allocator: std.mem.Allocator,
    input: *const [n]f32,
) ![]u8 {
    // Stage 1: Quantile Binning
    const bins = quantile.quantileBinning(n, Q, input);

    // Stage 2: Markov Transition Matrix
    var W = [_][Q]f32{[_]f32{0.0} ** Q} ** Q;
    
    // Accumulate transitions between adjacent time steps
    if (n > 1) {
        var i: usize = 0;
        while (i < n - 1) : (i += 1) {
            W[bins[i]][bins[i + 1]] += 1.0;
        }
    }
    
    // Row normalization (ensure stochastic matrix W)
    for (0..Q) |r| {
        var sum: f32 = 0.0;
        for (0..Q) |c| {
            sum += W[r][c];
        }
        if (sum > 0.0) {
            for (0..Q) |c| {
                W[r][c] /= sum;
            }
        }
    }
    
    // Stage 3 & 4: Upsampling QxQ to nxn and Rescaling to [0, 255]
    var out = try allocator.alloc(u8, n * n);
    errdefer allocator.free(out);
    
    if (n == 0) return out;
    if (n == 1) {
        // Safe mapping for n=1
        out[0] = @as(u8, @intFromFloat(W[bins[0]][bins[0]] * 255.0));
        return out;
    }
    
    for (0..n) |i| {
        for (0..n) |j| {
            // Map coordinates from (n x n) -> (Q x Q)
            // We use (Q - 1) and (n - 1) to perfectly align the interpolation bounds
            const src_i = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(Q - 1)) / @as(f32, @floatFromInt(n - 1));
            const src_j = @as(f32, @floatFromInt(j)) * @as(f32, @floatFromInt(Q - 1)) / @as(f32, @floatFromInt(n - 1));
            
            // Bilinear interpolation upper and lower bounds
            const r1 = @as(usize, @intFromFloat(@floor(src_i)));
            var r2 = r1 + 1;
            if (r2 >= Q) r2 = Q - 1;
            
            const c1 = @as(usize, @intFromFloat(@floor(src_j)));
            var c2 = c1 + 1;
            if (c2 >= Q) c2 = Q - 1;
            
            // Distance from bounding index
            const dr = src_i - @as(f32, @floatFromInt(r1));
            const dc = src_j - @as(f32, @floatFromInt(c1));
            
            // Interpolate rows first
            const val1 = W[r1][c1] * (1.0 - dr) + W[r2][c1] * dr;
            const val2 = W[r1][c2] * (1.0 - dr) + W[r2][c2] * dr;
            
            // Interpolate columns
            const val = val1 * (1.0 - dc) + val2 * dc;
            
            // Rescale to [0, 255] and clamp safely
            var clamp_val = val;
            if (clamp_val < 0.0) clamp_val = 0.0;
            if (clamp_val > 1.0) clamp_val = 1.0;
            
            out[i * n + j] = @as(u8, @intFromFloat(clamp_val * 255.0));
        }
    }
    
    return out;
}
