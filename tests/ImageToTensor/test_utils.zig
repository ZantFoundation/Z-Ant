const std = @import("std");
const zant = @import("zant");
const testing = std.testing;
const utils = zant.ImageToTensor.utils;
const pkgAllocator = zant.utils.allocator;

test "normalize - converts 0-255 to 0-1 range" {
    const allocator = pkgAllocator.allocator;

    // Create a small test image with known values
    var channels = try utils.ColorChannels.init(allocator, 4, 3);
    defer channels.deinit(allocator);

    // Set test values in channels
    channels.ch1[0] = 0; // Min value
    channels.ch1[1] = 128; // Mid value
    channels.ch1[2] = 255; // Max value
    channels.ch1[3] = 64;

    channels.ch2[0] = 255;
    channels.ch2[1] = 0;
    channels.ch2[2] = 128;
    channels.ch2[3] = 192;

    channels.ch3[0] = 64;
    channels.ch3[1] = 192;
    channels.ch3[2] = 32;
    channels.ch3[3] = 224;

    // Set dimensions for the channels
    channels.width = 2;
    channels.height = 2;

    // Create output tensor with shape [3, 2, 2]
    var output = try allocator.alloc([][]f32, 3);
    for (0..3) |i| {
        output[i] = try allocator.alloc([]f32, 2);
        for (0..2) |j| {
            output[i][j] = try allocator.alloc(f32, 2);
        }
    }
    defer {
        for (0..3) |i| {
            for (0..2) |j| {
                allocator.free(output[i][j]);
            }
            allocator.free(output[i]);
        }
        allocator.free(output);
    }

    // Call normalize function
    try utils.normalize(f32, &channels, output);

    // Check normalized values - [channel][row][col]
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0][0][0], 0.001); // ch1[0]
    try testing.expectApproxEqAbs(@as(f32, 0.502), output[0][0][1], 0.001); // ch1[1]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0][1][0], 0.001); // ch1[2]
    try testing.expectApproxEqAbs(@as(f32, 0.251), output[0][1][1], 0.001); // ch1[3]

    try testing.expectApproxEqAbs(@as(f32, 1.0), output[1][0][0], 0.001); // ch2[0]
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1][0][1], 0.001); // ch2[1]
    try testing.expectApproxEqAbs(@as(f32, 0.502), output[1][1][0], 0.001); // ch2[2]
    try testing.expectApproxEqAbs(@as(f32, 0.753), output[1][1][1], 0.001); // ch2[3]

    try testing.expectApproxEqAbs(@as(f32, 0.251), output[2][0][0], 0.001); // ch3[0]
    try testing.expectApproxEqAbs(@as(f32, 0.753), output[2][0][1], 0.001); // ch3[1]
    try testing.expectApproxEqAbs(@as(f32, 0.125), output[2][1][0], 0.001); // ch3[2]
    try testing.expectApproxEqAbs(@as(f32, 0.878), output[2][1][1], 0.001); // ch3[3]
}

test "normalizeSigned - converts 0-255 to -1-1 range" {
    const allocator = testing.allocator;

    // Create a small test image with known values
    var channels = try utils.ColorChannels.init(allocator, 4, 3);
    defer channels.deinit(allocator);

    // Set test values in channels
    channels.ch1[0] = 0; // Min value -> -1.0
    channels.ch1[1] = 128; // Mid value -> 0.0
    channels.ch1[2] = 255; // Max value -> 1.0
    channels.ch1[3] = 64; // 64 -> -0.5

    channels.ch2[0] = 255;
    channels.ch2[1] = 0;
    channels.ch2[2] = 128;
    channels.ch2[3] = 192;

    channels.ch3[0] = 64;
    channels.ch3[1] = 192;
    channels.ch3[2] = 32;
    channels.ch3[3] = 224;

    // Set dimensions for the channels
    channels.width = 2;
    channels.height = 2;

    // Create output tensor with shape [3, 2, 2]
    var output = try allocator.alloc([][]f32, 3);
    for (0..3) |i| {
        output[i] = try allocator.alloc([]f32, 2);
        for (0..2) |j| {
            output[i][j] = try allocator.alloc(f32, 2);
        }
    }
    defer {
        for (0..3) |i| {
            for (0..2) |j| {
                allocator.free(output[i][j]);
            }
            allocator.free(output[i]);
        }
        allocator.free(output);
    }

    // Call normalizeSigned function
    try utils.normalizeSigned(f32, &channels, output);

    // Check normalized values
    try testing.expectApproxEqAbs(@as(f32, -1.0), output[0][0][0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0][0][1], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0][1][0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -0.498), output[0][1][1], 0.001);

    try testing.expectApproxEqAbs(@as(f32, 1.0), output[1][0][0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -1.0), output[1][0][1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1][1][0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 0.506), output[1][1][1], 0.001);

    try testing.expectApproxEqAbs(@as(f32, -0.498), output[2][0][0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.506), output[2][0][1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -0.749), output[2][1][0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.757), output[2][1][1], 0.001);
}

test "normalize with f64 type" {
    const allocator = testing.allocator;

    // Create a small test image
    var channels = try utils.ColorChannels.init(allocator, 1, 3);
    defer channels.deinit(allocator);

    // Set test values
    channels.ch1[0] = 255;
    channels.ch2[0] = 128;
    channels.ch3[0] = 0;

    // Set dimensions
    channels.width = 1;
    channels.height = 1;

    // Create output tensor with shape [3, 1, 1]
    var output = try allocator.alloc([][]f64, 3);
    for (0..3) |i| {
        output[i] = try allocator.alloc([]f64, 1);
        for (0..1) |j| {
            output[i][j] = try allocator.alloc(f64, 1);
        }
    }
    defer {
        for (0..3) |i| {
            for (0..1) |j| {
                allocator.free(output[i][j]);
            }
            allocator.free(output[i]);
        }
        allocator.free(output);
    }

    // Call normalize function with f64 type
    try utils.normalize(f64, &channels, output);

    // Check normalized values with higher precision
    try testing.expectApproxEqAbs(@as(f64, 1.0), output[0][0][0], 0.00001);
    try testing.expectApproxEqAbs(@as(f64, 0.50196078431), output[1][0][0], 0.00001);
    try testing.expectApproxEqAbs(@as(f64, 0.0), output[2][0][0], 0.00001);
}
