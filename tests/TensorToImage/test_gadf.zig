const std = @import("std");
const testing = std.testing;

const zant = @import("zant");
const gadf = zant.TensorToImage.gadf.gadf;
const utils = zant.TensorToImage.gaf_utils;

const NormRange = utils.NormRange;

test "GADF Reference Python Test (pyts)" {
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8 };
    const n = input.len;

    // Output atteso generato dallo script pyts (method='difference', range=(-1, 1))
    const expected = [_]f32{ 0.0, 0.680239, -0.962091, 0.962091, -0.680239, 0.0, -0.890724, 0.890724, 0.962091, 0.890724, 0.0, 0.0, -0.962091, -0.890724, -0.0, 0.0 };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Calcoliamo la matrice usando l'intervallo [-1, 1] per allinearci a pyts
    const result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    try testing.expectEqual(@as(usize, n * n), result.len);

    // Check elemento per elemento usando una tolleranza float (epsilon)
    const epsilon: f32 = 1e-5;
    for (expected, result) |exp, res| {
        const is_approx_equal = std.math.approxEqAbs(f32, exp, res, epsilon);
        if (!is_approx_equal) {
            std.debug.print("\nMismatch! Expected: {d:.6}, Got: {d:.6}\n", .{ exp, res });
        }
        try testing.expect(is_approx_equal);
    }
}
