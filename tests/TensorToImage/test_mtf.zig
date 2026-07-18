const std = @import("std");
const testing = std.testing;

const TensorToImage = @import("TensorToImage");
const mtf_utils = TensorToImage.mtf.mtf_utils;
const mtf_core = TensorToImage.mtf.mtf;

test "quantileBins: basic 8-element 4-bin reference" {
    // sorted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
    // edges at idx k*8/4: k=1->idx2=0.3, k=2->idx4=0.5, k=3->idx6=0.8
    // expected bins: [0, 2, 1, 3, 0, 3, 1, 2]
    const input = [_]f32{ 0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6 };
    const expected = [_]usize{ 0, 2, 1, 3, 0, 3, 1, 2 };

    var sorted_buf: [8]f32 = undefined;
    var bins_out: [8]usize = undefined;

    mtf_utils.quantileBins(&input, 4, &sorted_buf, &bins_out);

    try testing.expectEqualSlices(usize, &expected, &bins_out);
}

test "quantileBins: flatline all bins valid" {
    const input = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    var sorted_buf: [4]f32 = undefined;
    var bins_out: [4]usize = undefined;

    mtf_utils.quantileBins(&input, 4, &sorted_buf, &bins_out);

    for (bins_out) |b| {
        try testing.expect(b < 4);
    }
}

test "quantileBins: Q=1 assigns all to bin 0" {
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var sorted_buf: [3]f32 = undefined;
    var bins_out: [3]usize = undefined;

    mtf_utils.quantileBins(&input, 1, &sorted_buf, &bins_out);

    for (bins_out) |b| {
        try testing.expectEqual(@as(usize, 0), b);
    }
}

test "transitionMatrix: known transitions from 8-element series" {
    // bins = [0, 2, 1, 3, 0, 3, 1, 2], Q=4
    // transitions: (0,2),(2,1),(1,3),(3,0),(0,3),(3,1),(1,2)
    // Row 0: [0, 0, 0.5, 0.5]   Row 1: [0, 0, 0.5, 0.5]
    // Row 2: [0, 1.0, 0, 0]     Row 3: [0.5, 0.5, 0, 0]
    const bins = [_]usize{ 0, 2, 1, 3, 0, 3, 1, 2 };
    const q: usize = 4;
    var matrix: [16]f32 = undefined;

    mtf_utils.transitionMatrix(&bins, q, &matrix);

    const eps: f32 = 1e-5;
    const expected = [_]f32{
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.5, 0.5,
        0.0, 1.0, 0.0, 0.0,
        0.5, 0.5, 0.0, 0.0,
    };
    for (expected, matrix) |exp, got| {
        try testing.expect(std.math.approxEqAbs(f32, exp, got, eps));
    }
}

test "transitionMatrix: rows sum to 1 or all-zero" {
    const bins = [_]usize{ 0, 1, 2, 1, 0 };
    const q: usize = 3;
    var matrix: [9]f32 = undefined;

    mtf_utils.transitionMatrix(&bins, q, &matrix);

    const eps: f32 = 1e-5;
    for (0..q) |r| {
        var sum: f32 = 0.0;
        for (0..q) |c| sum += matrix[r * q + c];
        try testing.expect(std.math.approxEqAbs(f32, sum, 1.0, eps) or
            std.math.approxEqAbs(f32, sum, 0.0, eps));
    }
}

test "transitionMatrix: dead row stays zero" {
    // bin 2 never appears — its row must be all-zero
    const bins = [_]usize{ 0, 1, 0, 1 };
    const q: usize = 3;
    var matrix: [9]f32 = [_]f32{0.0} ** 9;

    mtf_utils.transitionMatrix(&bins, q, &matrix);

    try testing.expectEqual(@as(f32, 0.0), matrix[2 * q + 0]);
    try testing.expectEqual(@as(f32, 0.0), matrix[2 * q + 1]);
    try testing.expectEqual(@as(f32, 0.0), matrix[2 * q + 2]);
}

test "lean_mtf: full pipeline reference" {
    // input = [0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6], Q=4, n=8
    // bins = [0, 2, 1, 3, 0, 3, 1, 2]
    // W (row-stochastic):
    //   Row 0: [0, 0, 0.5, 0.5]   Row 1: [0, 0, 0.5, 0.5]
    //   Row 2: [0, 1.0, 0, 0]     Row 3: [0.5, 0.5, 0, 0]
    // M[i][j] = W[bins[i]][bins[j]]
    const input = [_]f32{ 0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6 };
    const q: usize = 4;
    const expected = [_]f32{
        0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
        0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
        0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    };

    var sorted_buf: [8]f32 = undefined;
    var bins_buf: [8]usize = undefined;
    var matrix_buf: [16]f32 = undefined;
    var output: [64]f32 = undefined;

    mtf_core.lean_mtf(&input, q, &sorted_buf, &bins_buf, &matrix_buf, &output);

    const eps: f32 = 1e-5;
    for (expected, output) |exp, got| {
        try testing.expect(std.math.approxEqAbs(f32, exp, got, eps));
    }
}

test "mtf standard: output length is n*n" {
    const input = [_]f32{ 0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try mtf_core.mtf(arena.allocator(), &input, 4);
    try testing.expectEqual(n * n, result.len);
}

test "mtf standard: InputTooShort on single element" {
    const input = [_]f32{0.5};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = mtf_core.mtf(arena.allocator(), &input, 4);
    try testing.expectError(error.InputTooShort, result);
}

test "mtf standard: InvalidBins on q=0" {
    const input = [_]f32{ 0.1, 0.5, 0.3 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = mtf_core.mtf(arena.allocator(), &input, 0);
    try testing.expectError(error.InvalidBins, result);
}

test "mtf standard: all values in [0, 1]" {
    const input = [_]f32{ 1.0, 3.0, 2.0, 4.0, 1.5, 3.5 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try mtf_core.mtf(arena.allocator(), &input, 3);
    for (result) |val| {
        try testing.expect(val >= 0.0 and val <= 1.0);
    }
}
