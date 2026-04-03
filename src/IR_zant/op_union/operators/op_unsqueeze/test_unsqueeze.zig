const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_shape);

test "test unsqueeze axis out of bounds error" {
    tests_log.info("\n test: unsqueeze axis out of bounds error\n", .{});
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &inputShape);
    defer tensor.deinit();

    // 3 is not a valid axes {0, 1, 2} are valid
    var axesArray: [1]i64 = [_]i64{3};
    var axesShape: [1]usize = [_]usize{1};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.unsqueeze(f32, &tensor, &axesTensor));
}

// -------------------------------------------------------------
// Test for DuplicateAxis error
test "test unsqueeze duplicate axis error" {
    tests_log.info("\n test: unsqueeze duplicate axis error\n", .{});
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &inputShape);
    defer tensor.deinit();

    // Axes tensor contains a dupliacate
    var axesArray: [2]i64 = [_]i64{ 0, 0 };
    var axesShape: [1]usize = [_]usize{2};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    try std.testing.expectError(TensorError.DuplicateAxis, TensMath.unsqueeze(f32, &tensor, &axesTensor));
}

test "unsqueeze - basic" {
    tests_log.info("\n     test: unsqueeze - basic", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Add dimension at axis 1
    var axesArray: [1]i64 = [_]i64{1};
    var axesShape: [1]usize = [_]usize{1};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    var result = try TensMath.unsqueeze(f32, &tensor, &axesTensor);
    defer result.deinit();

    // Expected shape: [2, 1, 3]
    try std.testing.expect(result.shape.len == 3);
    try std.testing.expect(result.shape[0] == 2);
    try std.testing.expect(result.shape[1] == 1);
    try std.testing.expect(result.shape[2] == 3);

    // Data should be preserved
    for (0..6) |i| {
        try std.testing.expect(result.data[i] == tensor.data[i]);
    }
}

test "unsqueeze - multiple axes" {
    tests_log.info("\n     test: unsqueeze - multiple axes", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x2 tensor
    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 2 };
    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Add dimensions at axes 0 and 2
    var axesArray: [2]i64 = [_]i64{ 0, 2 };
    var axesShape: [1]usize = [_]usize{2};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    var result = try TensMath.unsqueeze(f32, &tensor, &axesTensor);
    defer result.deinit();

    // Expected shape: [1, 2, 1, 2]
    try std.testing.expect(result.shape.len == 4);
    try std.testing.expect(result.shape[0] == 1);
    try std.testing.expect(result.shape[1] == 2);
    try std.testing.expect(result.shape[2] == 1);
    try std.testing.expect(result.shape[3] == 2);

    // Data should be preserved
    for (0..4) |i| {
        try std.testing.expect(result.data[i] == tensor.data[i]);
    }
}

test "unsqueeze - negative axes" {
    tests_log.info("\n     test: unsqueeze - negative axes", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Add dimension at axis -1 (equivalent to 2)
    var axesArray: [1]i64 = [_]i64{-1};
    var axesShape: [1]usize = [_]usize{1};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    var result = try TensMath.unsqueeze(f32, &tensor, &axesTensor);
    defer result.deinit();

    // Expected shape: [2, 3, 1]
    try std.testing.expect(result.shape.len == 3);
    try std.testing.expect(result.shape[0] == 2);
    try std.testing.expect(result.shape[1] == 3);
    try std.testing.expect(result.shape[2] == 1);

    // Data should be preserved
    for (0..6) |i| {
        try std.testing.expect(result.data[i] == tensor.data[i]);
    }
}

test "unsqueeze - scalar tensor" {
    tests_log.info("\n     test: unsqueeze - scalar tensor", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 1x1 tensor
    var inputArray: [1][1]f32 = [_][1]f32{[_]f32{42.0}};
    var shape: [2]usize = [_]usize{ 1, 1 };
    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Add dimensions at multiple positions
    var axesArray: [3]i64 = [_]i64{ 0, 2, 3 };
    var axesShape: [1]usize = [_]usize{3};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    var result = try TensMath.unsqueeze(f32, &tensor, &axesTensor);
    defer result.deinit();

    // Expected shape: [1, 1, 1, 1, 1]
    try std.testing.expect(result.shape.len == 5);
    for (result.shape) |dim| {
        try std.testing.expect(dim == 1);
    }

    // Data should be preserved
    try std.testing.expect(result.data[0] == 42.0);
}

test "unsqueeze - error cases" {
    tests_log.info("\n     test: unsqueeze - error cases", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x2 tensor
    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 2 };
    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Test 1: Invalid axis (too large)
    {
        var axesArray: [1]i64 = [_]i64{5};
        var axesShape: [1]usize = [_]usize{1};
        var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
        defer axesTensor.deinit();

        try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.unsqueeze(f32, &tensor, &axesTensor));
    }

    // Test 2: Invalid axis (too negative)
    {
        var axesArray: [1]i64 = [_]i64{-5};
        var axesShape: [1]usize = [_]usize{1};
        var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
        defer axesTensor.deinit();

        try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.unsqueeze(f32, &tensor, &axesTensor));
    }
}
