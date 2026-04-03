const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "Flatten - Basic" {
    tests_log.info("\n     test: Flatten - Basic ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = 1
    var flattened = try TensMath.flatten(u8, &tensor, 1);
    defer flattened.deinit();

    // Verify shape: [2, 3]
    try std.testing.expect(flattened.shape[0] == 2);
    try std.testing.expect(flattened.shape[1] == 3);
    try std.testing.expect(flattened.size == 6);

    // Verify data preservation
    try std.testing.expect(flattened.data[0] == 1);
    try std.testing.expect(flattened.data[1] == 2);
    try std.testing.expect(flattened.data[2] == 3);
    try std.testing.expect(flattened.data[3] == 4);
    try std.testing.expect(flattened.data[4] == 5);
    try std.testing.expect(flattened.data[5] == 6);
}

test "Flatten - Multi-dimensional" {
    tests_log.info("\n     test: Flatten - Multi-dimensional ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3x4 tensor
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = 1
    var flattened = try TensMath.flatten(u8, &tensor, 1);
    defer flattened.deinit();

    // Verify shape: [2, 12]
    try std.testing.expect(flattened.shape[0] == 2);
    try std.testing.expect(flattened.shape[1] == 12);
    try std.testing.expect(flattened.size == 24);

    // Verify data preservation
    for (0..24) |i| {
        try std.testing.expect(flattened.data[i] == i + 1);
    }
}

test "Flatten - Negative axis" {
    tests_log.info("\n     test: Flatten - Negative axis ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3x4 tensor
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = -2 (equivalente a axis = 1)
    var flattened = try TensMath.flatten(u8, &tensor, -2);
    defer flattened.deinit();

    // Verify shape: [2, 12]
    try std.testing.expect(flattened.shape[0] == 2);
    try std.testing.expect(flattened.shape[1] == 12);
    try std.testing.expect(flattened.size == 24);

    // Verify data preservation
    for (0..24) |i| {
        try std.testing.expect(flattened.data[i] == i + 1);
    }
}

test "Flatten - Axis zero" {
    tests_log.info("\n     test: Flatten - Axis zero ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3x4 tensor
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = 0
    var flattened = try TensMath.flatten(u8, &tensor, 0);
    defer flattened.deinit();

    // Verify shape: [1, 24]
    try std.testing.expect(flattened.shape[0] == 1);
    try std.testing.expect(flattened.shape[1] == 24);
    try std.testing.expect(flattened.size == 24);

    // Verify data preservation
    for (0..24) |i| {
        try std.testing.expect(flattened.data[i] == i + 1);
    }
}

test "Flatten - Axis equal to rank" {
    tests_log.info("\n     test: Flatten - Axis equal to rank ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3x4 tensor
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = 3
    var flattened = try TensMath.flatten(u8, &tensor, 3);
    defer flattened.deinit();

    // Verify shape: [24, 1]
    try std.testing.expect(flattened.shape[0] == 24);
    try std.testing.expect(flattened.shape[1] == 1);
    try std.testing.expect(flattened.size == 24);

    // Verify data preservation
    for (0..24) |i| {
        try std.testing.expect(flattened.data[i] == i + 1);
    }
}

test "Flatten - Scalar tensor" {
    tests_log.info("\n     test: Flatten - Scalar tensor ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a scalar tensor
    var inputArray: [1]u8 = [_]u8{42};
    var shape: [0]usize = [_]usize{};

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Flatten con axis = 0
    var flattened = try TensMath.flatten(u8, &tensor, 0);
    defer flattened.deinit();

    // Verify shape: [1, 1]
    try std.testing.expect(flattened.shape[0] == 1);
    try std.testing.expect(flattened.shape[1] == 1);
    try std.testing.expect(flattened.size == 1);

    // Verify data preservation
    try std.testing.expect(flattened.data[0] == 42);
}

test "Flatten - Empty tensor" {
    tests_log.info("\n     test: Flatten - Empty tensor ", .{});
    const allocator = pkgAllocator.allocator;

    // Create an empty tensor: [0, 3, 4]
    var shape: [3]usize = [_]usize{ 0, 3, 4 };
    var tensor = try Tensor(u8).fromShape(&allocator, &shape);
    defer tensor.deinit();

    // Flatten con axis = 1
    var flattened = try TensMath.flatten(u8, &tensor, 1);
    defer flattened.deinit();

    // Verify shape: [0, 12]
    try std.testing.expect(flattened.shape[0] == 0);
    try std.testing.expect(flattened.shape[1] == 12);
    try std.testing.expect(flattened.size == 0);

    // Verify no data
    try std.testing.expect(flattened.data.len == 0);
}

test "Flatten - Invalid axis" {
    tests_log.info("\n     test: Flatten - Invalid axis ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3x4 tensor
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Try with invalid axis (4)
    try std.testing.expectError(TensorMathError.AxisOutOfRange, TensMath.flatten(u8, &tensor, 4));

    // Try with invalid axis (-4)
    try std.testing.expectError(TensorMathError.AxisOutOfRange, TensMath.flatten(u8, &tensor, -4));
}
