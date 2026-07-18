const std = @import("std");
const testing = std.testing;

const TensorToImage = @import("TensorToImage");
const utils = TensorToImage.gaf_utils;
const normalize = utils.normalize;
const NormRange = utils.NormRange;
const NormalizeError = utils.NormalizeError;

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

test "normalize: empty input returns EmptyInput error" {
    const input = [_]f32{};
    var output = [_]f32{};
    try testing.expectError(NormalizeError.EmptyInput, normalize(&input, &output, .MinusOneToOne));
}

test "normalize: mismatched lengths returns LengthMismatch error" {
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0 };
    try testing.expectError(NormalizeError.LengthMismatch, normalize(&input, &output, .MinusOneToOne));
}

// ---------------------------------------------------------------------------
// MinusOneToOne
// ---------------------------------------------------------------------------

test "normalize MinusOneToOne: min maps to -1, max maps to +1" {
    const input = [_]f32{ 1.0, 3.0, 5.0 };
    var output: [3]f32 = undefined;
    try normalize(&input, &output, .MinusOneToOne);

    try testing.expectApproxEqAbs(@as(f32, -1.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[2], 1e-6);
}

test "normalize MinusOneToOne: all values in [-1, 1]" {
    const input = [_]f32{ 3.7, -2.1, 0.0, 100.0, -50.0 };
    var output: [5]f32 = undefined;
    try normalize(&input, &output, .MinusOneToOne);

    for (output) |v| {
        try testing.expect(v >= -1.0 and v <= 1.0);
    }
}

test "normalize MinusOneToOne: flat series produces all zeros" {
    // range = 0, division avoided — all outputs must be 0.0
    const input = [_]f32{ 7.0, 7.0, 7.0, 7.0 };
    var output: [4]f32 = undefined;
    try normalize(&input, &output, .MinusOneToOne);

    for (output) |v| {
        try testing.expectEqual(@as(f32, 0.0), v);
    }
}

// ---------------------------------------------------------------------------
// ZeroToOne
// ---------------------------------------------------------------------------

test "normalize ZeroToOne: min maps to 0, max maps to 1" {
    const input = [_]f32{ 2.0, 4.0, 6.0 };
    var output: [3]f32 = undefined;
    try normalize(&input, &output, .ZeroToOne);

    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[2], 1e-6);
}

test "normalize ZeroToOne: all values in [0, 1]" {
    const input = [_]f32{ -99.0, 0.5, 12.3, 1e6, -1e6 };
    var output: [5]f32 = undefined;
    try normalize(&input, &output, .ZeroToOne);

    for (output) |v| {
        try testing.expect(v >= 0.0 and v <= 1.0);
    }
}

test "normalize ZeroToOne: flat series produces all zeros" {
    const input = [_]f32{ 3.0, 3.0, 3.0 };
    var output: [3]f32 = undefined;
    try normalize(&input, &output, .ZeroToOne);

    for (output) |v| {
        try testing.expectEqual(@as(f32, 0.0), v);
    }
}

// ---------------------------------------------------------------------------
// Single-element input
// ---------------------------------------------------------------------------

test "normalize: single element treated as flat series, outputs 0" {
    const input = [_]f32{42.0};
    var output: [1]f32 = undefined;
    try normalize(&input, &output, .MinusOneToOne);
    try testing.expectEqual(@as(f32, 0.0), output[0]);
}
