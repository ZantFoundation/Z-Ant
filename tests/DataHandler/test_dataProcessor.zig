const std = @import("std");
const zant = @import("zant");
const DataProc = zant.data_handler.data_processor;
const NormalizType = DataProc.NormalizationType;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;

test "normalize float" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    for (t1.data) |*val| {
        try std.testing.expect(1.0 >= val.*);
        try std.testing.expect(0.0 <= val.*);
    }
}

test "normalize float all different" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 3.0, 10.0 },
        [_]f32{ 4.0, 5.0, 1.0, 2.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 4 }; // 2x4 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    t1.info();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    t1.info();

    try std.testing.expect(1.0 - t1.data[3] < 0.001);
    try std.testing.expect(1.0 - t1.data[5] < 0.001);

    for (t1.data) |*val| {
        try std.testing.expect(1.0 >= val.*);
        try std.testing.expect(0.0 <= val.*);
    }
}

test "normalize delta 0 " {
    std.debug.print("\n     test: normalize delta 0", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 1.0 },
        [_]f32{ 0.0, 0.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    for (t1.data) |*val| {
        try std.testing.expect(0 == val.*);
    }
}

// Helper function to compute mean of a slice
fn compute_mean(comptime T: anytype, data: []const T) f64 {
    var sum: f64 = 0.0;
    for (data) |val| {
        sum += @as(f64, val);
    }
    return sum / @as(f64, @floatFromInt(data.len));
}

// Helper function to compute variance of a slice given its mean
fn compute_variance(comptime T: anytype, data: []const T, mean: f64) f64 {
    var sum_sq_diff: f64 = 0.0;
    for (data) |val| {
        const diff = @as(f64, val) - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / @as(f64, @floatFromInt(data.len));
}

test "standard normalize float 2D" {
    std.debug.print("\n     test: standard normalize float 2D", .{});

    const allocator = pkgAllocator.allocator;

    // Define a 2x2 tensor with diverse float values
    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    // Perform standard normalization
    try DataProc.normalize(f32, &t1, NormalizType.StandardDeviationNormalization);

    // Check each slice (since it's 2D, each row is a slice)
    for (0..t1.shape[0]) |b| {
        const slice_offset = b * t1.shape[1];
        const slice_size = t1.shape[1];
        const slice_data = t1.data[slice_offset .. slice_offset + slice_size];

        // Compute mean and variance
        var sum: f64 = 0.0;
        for (slice_data) |val| {
            sum += @as(f64, val);
        }
        const mean = sum / @as(f64, @floatFromInt(slice_size));

        var sum_sq_diff: f64 = 0.0;
        for (slice_data) |val| {
            const diff = @as(f64, val) - mean;
            sum_sq_diff += diff * diff;
        }

        // Expect mean close to 0 and variance close to 1
        var normalized_sum: f64 = 0.0;
        var normalized_sq_sum: f64 = 0.0;
        for (slice_data) |val| {
            normalized_sum += @as(f64, val);
            normalized_sq_sum += @as(f64, val) * @as(f64, val);
        }
        const normalized_mean = normalized_sum / @as(f64, @floatFromInt(slice_size));
        const normalized_variance = normalized_sq_sum / @as(f64, @floatFromInt(slice_size));

        try std.testing.expect(@abs(normalized_mean) < 1e-4);
        try std.testing.expect(@abs(normalized_variance - 1.0) < 1e-4);
    }
}

test "normalize 4D tensor known values" {
    std.debug.print("\n     test: normalize 4D tensor known values", .{});

    const allocator = pkgAllocator.allocator;

    // Define a 1x1x2x2 tensor with known float values
    var inputArray: [1][1][2][2]f32 = [1][1][2][2]f32{
        .{
            .{
                .{
                    1.0, 2.0, // Width 0, Height 0 and 1
                },
                .{
                    3.0, 4.0, // Width 1, Height 0 and 1
                },
            },
        },
    };

    var shape: [4]usize = [_]usize{ 1, 1, 2, 2 }; // Shape: [1, 1, 2, 2]

    // Initialize the tensor from the input array
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    // t1.info(); // Optional: Print tensor info for debugging

    // Perform min-max normalization
    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    // Optional: Print tensor data after normalization for debugging
    // print_tensor(f32, &t1);

    // Expected normalized tensor:
    // [
    //   [
    //     [ [0.0, 0.3333], [0.6667, 1.0] ]
    //   ]
    // ]

    // Tolerance for floating point comparison
    const tol: f32 = 1e-3;

    // Batch 0, Channel 0, Width 0, Height 0: 1.0 -> 0.0
    std.debug.print("t1.data[0]: {any}", .{t1.data[0]});
    try std.testing.expect(@abs(t1.data[0] - 0.0) < tol);
    std.debug.print("t1.data[0]: {any}", .{t1.data[0]});
    // Batch 0, Channel 0, Width 0, Height 1: 2.0 -> ~0.3333
    try std.testing.expect(@abs(t1.data[1] - 0.3333) < tol);
    std.debug.print("t1.data[1]: {any}", .{t1.data[1]});
    // Batch 0, Channel 0, Width 1, Height 0: 3.0 -> ~0.6667
    std.debug.print("t1.data[2]: {any}", .{t1.data[2]});
    try std.testing.expect(@abs(t1.data[2] - 0.6667) < tol);
    // Batch 0, Channel 0, Width 1, Height 1: 4.0 -> 1.0
    std.debug.print("t1.data[3]: {any}", .{t1.data[3]});
    try std.testing.expect(@abs(t1.data[3] - 1.0) < tol);
}
