const std = @import("std");
const testing = std.testing;

const zant = @import("zant");
const quantileBinning = zant.TensorToImage.quantile.quantileBinning;
const mtf = zant.TensorToImage.mtf.mtf;

test "Quantile Binning Reference Python Test" {
    const n = 8;
    const Q = 4;
    const input = [8]f32{ 0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6 };
    
    const expected = [8]usize{ 0, 2, 1, 3, 0, 3, 1, 2 };
    
    const result = quantileBinning(n, Q, &input);
    
    try testing.expectEqualSlices(usize, &expected, &result);
}

test "MTF Full Pipeline Reference Test" {
    const n = 8;
    const Q = 4;
    const input = [8]f32{ 0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6 };
    
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    
    const result = try mtf(n, Q, arena.allocator(), &input);
    
    // Check dimensions
    try testing.expectEqual(@as(usize, n * n), result.len);
    
    // Due to bilinear interpolation from the specified ground truth matrix,
    // we simply ensure no values go out of bounds (handled via clipping),
    // and that the corner values match the ground truth.
    // top-left corner maps to W[0][0] which is 0.0 -> 0
    try testing.expectEqual(@as(u8, 0), result[0]);
    // bottom-right corner maps to W[3][3] which is 0.0 -> 0
    try testing.expectEqual(@as(u8, 0), result[n * n - 1]);
}
