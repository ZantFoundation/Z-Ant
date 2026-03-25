const std = @import("std");

/// Computes quantile bin assignments for a given time series.
/// 'n' is the time series length. 'Q' is the number of quantile bins.
/// Returns an array of n bin assignments in the range [0, Q-1].
pub fn quantileBinning(
    comptime n: usize,
    comptime Q: usize,
    input: *const [n]f32,
) [n]usize {
    // If n is 0 or 1, handle gracefully
    if (n == 0) return [0]usize{};
    if (n == 1) return [1]usize{0};

    // Make a mutable local copy to sort
    var sorted_input = input.*;
    std.mem.sort(f32, &sorted_input, {}, std.sort.asc(f32));

    var bins: [n]usize = undefined;
    for (input.*, 0..) |val, i| {
        // Find rank (index in sorted array)
        // If duplicates exist, indexOfScalar returns the first index.
        const rank = std.mem.indexOfScalar(f32, &sorted_input, val) orelse 0;
        
        // Compute bin index
        var q_i = @as(usize, @intFromFloat(@as(f32, @floatFromInt(rank)) / @as(f32, @floatFromInt(n)) * @as(f32, @floatFromInt(Q))));
        
        // Clamp to max [0, Q-1]
        if (q_i >= Q) {
            q_i = Q - 1;
        }
        
        bins[i] = q_i;
    }
    
    return bins;
}
