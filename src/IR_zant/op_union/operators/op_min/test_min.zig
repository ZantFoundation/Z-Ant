const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_min);

// ---------------------------------------------------------------
// -------------------- TESTS FOR MIN OPERATOR -------------------
// ---------------------------------------------------------------

// Test basic two-tensor min operation
test "Min - basic two tensors" {
    tests_log.info("\n     test: Min - basic two tensors\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [3]usize = [_]usize{ 2, 2, 2 };
    var dataA: [8]f32 = [_]f32{ 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0 };
    var dataB: [8]f32 = [_]f32{ 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0, 2.0 };
    const expected: [8]f32 = [_]f32{ 1.0, 4.0, 3.0, 1.0, 2.0, 5.0, 4.0, 2.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("Basic two-tensor min test passed\n", .{});
}

// Test min with multiple tensors (3 tensors)
test "Min - multiple tensors (3)" {
    tests_log.info("\n     test: Min - multiple tensors (3)\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 3 };
    var dataA: [6]f32 = [_]f32{ 5.0, 3.0, 8.0, 2.0, 7.0, 4.0 };
    var dataB: [6]f32 = [_]f32{ 4.0, 6.0, 2.0, 9.0, 1.0, 5.0 };
    var dataC: [6]f32 = [_]f32{ 3.0, 5.0, 7.0, 1.0, 6.0, 3.0 };
    const expected: [6]f32 = [_]f32{ 3.0, 3.0, 2.0, 1.0, 1.0, 3.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();
    var tensorC = try Tensor(f32).fromArray(&allocator, &dataC, &shape);
    defer tensorC.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    var inputs = [_]*Tensor(f32){ &tensorA, &tensorB, &tensorC };
    try TensMath.min_lean(f32, &inputs, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("Multiple tensors min test passed\n", .{});
}

// Test min with all positive values
test "Min - all positive values" {
    tests_log.info("\n     test: Min - all positive values\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{5};
    var dataA: [5]f32 = [_]f32{ 10.0, 20.0, 5.0, 15.0, 8.0 };
    var dataB: [5]f32 = [_]f32{ 12.0, 18.0, 6.0, 14.0, 9.0 };
    const expected: [5]f32 = [_]f32{ 10.0, 18.0, 5.0, 14.0, 8.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("All positive values min test passed\n", .{});
}

// Test min with all negative values
test "Min - all negative values" {
    tests_log.info("\n     test: Min - all negative values\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{4};
    var dataA: [4]f32 = [_]f32{ -5.0, -2.0, -8.0, -3.0 };
    var dataB: [4]f32 = [_]f32{ -6.0, -1.0, -7.0, -4.0 };
    const expected: [4]f32 = [_]f32{ -6.0, -2.0, -8.0, -4.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("All negative values min test passed\n", .{});
}

// Test min with mixed positive and negative values
test "Min - mixed positive and negative" {
    tests_log.info("\n     test: Min - mixed positive and negative\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{6};
    var dataA: [6]f32 = [_]f32{ -5.0, 3.0, -2.0, 7.0, 0.0, -1.0 };
    var dataB: [6]f32 = [_]f32{ -3.0, 5.0, -4.0, 2.0, 1.0, -2.0 };
    const expected: [6]f32 = [_]f32{ -5.0, 3.0, -4.0, 2.0, 0.0, -2.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("Mixed positive and negative min test passed\n", .{});
}

// Test min with zeros
test "Min - with zero values" {
    tests_log.info("\n     test: Min - with zero values\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{5};
    var dataA: [5]f32 = [_]f32{ 0.0, 5.0, 0.0, -3.0, 2.0 };
    var dataB: [5]f32 = [_]f32{ 1.0, 0.0, -1.0, 0.0, 0.0 };
    const expected: [5]f32 = [_]f32{ 0.0, 0.0, -1.0, -3.0, 0.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("Min with zero values test passed\n", .{});
}

// Test min with identical tensors
test "Min - identical tensors" {
    tests_log.info("\n     test: Min - identical tensors\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 3 };
    var data: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &data, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &data, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(data[i], val, 1e-6);
    }

    tests_log.info("Identical tensors min test passed\n", .{});
}

// Test reduce_min (global minimum)
test "Min - reduce_min global" {
    tests_log.info("\n     test: Min - reduce_min global\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [3]usize = [_]usize{ 2, 2, 2 };
    var data: [8]f32 = [_]f32{ 5.0, 3.0, 8.0, -2.0, 7.0, 4.0, 1.0, 6.0 };

    var tensor = try Tensor(f32).fromArray(&allocator, &data, &shape);
    defer tensor.deinit();

    var result = try TensMath.reduce_min(f32, &tensor, null, false);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.size);
    try std.testing.expectApproxEqAbs(-2.0, result.data[0], 1e-6);

    tests_log.info("Global reduce_min test passed\n", .{});
}

// Test reduce_min with keepdims=true
test "Min - reduce_min with keepdims" {
    tests_log.info("\n     test: Min - reduce_min with keepdims\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 3, 2 };
    var data: [6]f32 = [_]f32{ 5.0, -1.0, 3.0, 7.0, 2.0, 4.0 };

    var tensor = try Tensor(f32).fromArray(&allocator, &data, &shape);
    defer tensor.deinit();

    var result = try TensMath.reduce_min(f32, &tensor, null, true);
    defer result.deinit();

    // With keepdims, shape should be [1, 1]
    try std.testing.expectEqual(@as(usize, 2), result.shape.len);
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    try std.testing.expectApproxEqAbs(-1.0, result.data[0], 1e-6);

    tests_log.info("Reduce_min with keepdims test passed\n", .{});
}

// Test min with single element tensors
test "Min - single element tensors" {
    tests_log.info("\n     test: Min - single element tensors\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{1};
    var dataA: [1]f32 = [_]f32{5.0};
    var dataB: [1]f32 = [_]f32{3.0};

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    try std.testing.expectApproxEqAbs(3.0, output.data[0], 1e-6);

    tests_log.info("Single element tensors min test passed\n", .{});
}

// Test min with integer type
test "Min - integer type (i32)" {
    tests_log.info("\n     test: Min - integer type (i32)\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{5};
    var dataA: [5]i32 = [_]i32{ 10, -5, 3, 7, -2 };
    var dataB: [5]i32 = [_]i32{ 8, -3, 5, 2, -4 };
    const expected: [5]i32 = [_]i32{ 8, -5, 3, 2, -4 };

    var tensorA = try Tensor(i32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(i32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(i32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(i32, &tensorA, &tensorB, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectEqual(expected[i], val);
    }

    tests_log.info("Integer type min test passed\n", .{});
}

// Test min with large tensors
test "Min - large tensors" {
    tests_log.info("\n     test: Min - large tensors\n", .{});

    const allocator = pkgAllocator.allocator;

    const size = 1000;
    var shape: [1]usize = [_]usize{size};

    var tensorA = try Tensor(f32).fromShape(&allocator, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromShape(&allocator, &shape);
    defer tensorB.deinit();

    // Fill with predictable values
    for (0..size) |i| {
        tensorA.data[i] = @as(f32, @floatFromInt(i));
        tensorB.data[i] = @as(f32, @floatFromInt(size - i));
    }

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    // Verify min is correctly computed
    for (0..size) |i| {
        const expected = @min(tensorA.data[i], tensorB.data[i]);
        try std.testing.expectApproxEqAbs(expected, output.data[i], 1e-6);
    }

    tests_log.info("Large tensors min test passed\n", .{});
}

// Test min with 4+ tensors
test "Min - four tensors" {
    tests_log.info("\n     test: Min - four tensors\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{4};
    var dataA: [4]f32 = [_]f32{ 5.0, 2.0, 8.0, 3.0 };
    var dataB: [4]f32 = [_]f32{ 4.0, 6.0, 1.0, 9.0 };
    var dataC: [4]f32 = [_]f32{ 3.0, 5.0, 7.0, 2.0 };
    var dataD: [4]f32 = [_]f32{ 6.0, 1.0, 4.0, 8.0 };
    const expected: [4]f32 = [_]f32{ 3.0, 1.0, 1.0, 2.0 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();
    var tensorC = try Tensor(f32).fromArray(&allocator, &dataC, &shape);
    defer tensorC.deinit();
    var tensorD = try Tensor(f32).fromArray(&allocator, &dataD, &shape);
    defer tensorD.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    var inputs = [_]*Tensor(f32){ &tensorA, &tensorB, &tensorC, &tensorD };
    try TensMath.min_lean(f32, &inputs, &output);

    for (output.data, 0..) |val, i| {
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-6);
    }

    tests_log.info("Four tensors min test passed\n", .{});
}

// Test get_min_output_shape
test "Min - get_min_output_shape" {
    tests_log.info("\n     test: Min - get_min_output_shape\n", .{});

    var shape1: [2]usize = [_]usize{ 3, 4 };
    var shape2: [2]usize = [_]usize{ 3, 4 };

    const input_shapes = [_][]const usize{ &shape1, &shape2 };
    const output_shape = try TensMath.get_min_output_shape(&input_shapes);
    defer pkgAllocator.allocator.free(output_shape);

    try std.testing.expectEqual(@as(usize, 2), output_shape.len);
    try std.testing.expectEqual(@as(usize, 3), output_shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output_shape[1]);

    tests_log.info("get_min_output_shape test passed\n", .{});
}

// Test min with extreme values (infinity, very large/small numbers)
test "Min - extreme values" {
    tests_log.info("\n     test: Min - extreme values\n", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [1]usize = [_]usize{4};
    var dataA: [4]f32 = [_]f32{ std.math.inf(f32), -std.math.inf(f32), 1e20, -1e20 };
    var dataB: [4]f32 = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const expected: [4]f32 = [_]f32{ 1.0, -std.math.inf(f32), 1.0, -1e20 };

    var tensorA = try Tensor(f32).fromArray(&allocator, &dataA, &shape);
    defer tensorA.deinit();
    var tensorB = try Tensor(f32).fromArray(&allocator, &dataB, &shape);
    defer tensorB.deinit();

    var output = try Tensor(f32).fromShape(&allocator, &shape);
    defer output.deinit();

    try TensMath.min_two_lean(f32, &tensorA, &tensorB, &output);

    try std.testing.expectApproxEqAbs(expected[0], output.data[0], 1e-6);
    try std.testing.expect(std.math.isNegativeInf(output.data[1]));
    try std.testing.expectApproxEqAbs(expected[2], output.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(expected[3], output.data[3], 1e10);

    tests_log.info("Extreme values min test passed\n", .{});
}
