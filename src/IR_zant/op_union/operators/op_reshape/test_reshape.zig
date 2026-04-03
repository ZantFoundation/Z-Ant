const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_shape);

test "Reshape" {
    tests_log.info("\n     test: Reshape ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 4 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [4]usize = [_]usize{ 1, 1, 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var new_shape: [4]isize = [_]isize{ 1, 1, 3, 2 };
    var new_tens = try TensMath.reshape(u8, &tensor, &new_shape, false);
    defer new_tens.deinit();

    tests_log.debug(" \n\n  new_tens.shape= {any} ", .{new_tens.shape});

    try std.testing.expect(new_tens.size == new_tens.size);
    try std.testing.expect(new_tens.shape[2] == 3);
    try std.testing.expect(new_tens.shape[3] == 2);
}

test "Reshape - Basic" {
    tests_log.info("\n     test: Reshape - Basic ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Reshape to 3x2
    var new_shape: [2]isize = [_]isize{ 3, 2 };
    var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
    defer reshaped.deinit();

    // Verify shape
    try std.testing.expect(reshaped.shape[0] == 3);
    try std.testing.expect(reshaped.shape[1] == 2);
    try std.testing.expect(reshaped.size == 6);

    // Verify data is preserved in row-major order
    try std.testing.expect(reshaped.data[0] == 1);
    try std.testing.expect(reshaped.data[1] == 2);
    try std.testing.expect(reshaped.data[2] == 3);
    try std.testing.expect(reshaped.data[3] == 4);
    try std.testing.expect(reshaped.data[4] == 5);
    try std.testing.expect(reshaped.data[5] == 6);
}

// test "Reshape - Basic Quantized" {
//     std.debug.print("\n     test: Reshape - Basic Quantized ", .{});
//     const allocator = pkgAllocator.allocator;

//     // Create a 2x3 tensor
//     var inputArray: [2][3]u8 = [_][3]u8{
//         [_]u8{ 1, 2, 3 },
//         [_]u8{ 4, 5, 6 },
//     };
//     var shape: [2]usize = [_]usize{ 2, 3 };

//     var tensor = try Tensor(u8).fromArrayQuantized(&allocator, &inputArray, &shape, 0.005, u8, 2);
//     defer tensor.deinit();

//     // Reshape to 3x2
//     var new_shape: [2]isize = [_]isize{ 3, 2 };
//     var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
//     defer reshaped.deinit();

//     // Verify shape
//     try std.testing.expect(reshaped.shape[0] == 3);
//     try std.testing.expect(reshaped.shape[1] == 2);
//     try std.testing.expect(reshaped.size == 6);

//     // Verify data is preserved in row-major order
//     try std.testing.expect(reshaped.data[0] == 1);
//     try std.testing.expect(reshaped.data[1] == 2);
//     try std.testing.expect(reshaped.data[2] == 3);
//     try std.testing.expect(reshaped.data[3] == 4);
//     try std.testing.expect(reshaped.data[4] == 5);
//     try std.testing.expect(reshaped.data[5] == 6);
//     try std.testing.expect(try reshaped.get_zero_point() == 2);
// }

test "Reshape - Multi-dimensional" {
    tests_log.info("\n     test: Reshape - Multi-dimensional ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x2x2 tensor
    var inputArray: [2][2][2]u8 = [_][2][2]u8{
        [_][2]u8{
            [_]u8{ 1, 2 },
            [_]u8{ 3, 4 },
        },
        [_][2]u8{
            [_]u8{ 5, 6 },
            [_]u8{ 7, 8 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 2, 2 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Reshape to 2x4
    var new_shape: [2]isize = [_]isize{ 2, 4 };
    var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
    defer reshaped.deinit();

    // Verify shape
    try std.testing.expect(reshaped.shape[0] == 2);
    try std.testing.expect(reshaped.shape[1] == 4);
    try std.testing.expect(reshaped.size == 8);

    // Verify data preservation
    for (0..8) |i| {
        try std.testing.expect(reshaped.data[i] == i + 1);
    }
}

test "Reshape - Error case" {
    tests_log.info("\n     test: Reshape - Error case ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Try to reshape to invalid size (2x4)
    var invalid_shape: [2]isize = [_]isize{ 2, 4 };
    try std.testing.expectError(TensorError.InputArrayWrongSize, TensMath.reshape(u8, &tensor, &invalid_shape, null));
}

test "Reshape - Same size different dimensions" {
    tests_log.info("\n     test: Reshape - Same size different dimensions ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Reshape to 1x6
    var new_shape: [2]isize = [_]isize{ 1, 6 };
    var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
    defer reshaped.deinit();

    // Verify shape
    try std.testing.expect(reshaped.shape[0] == 1);
    try std.testing.expect(reshaped.shape[1] == 6);
    try std.testing.expect(reshaped.size == 6);

    // Verify data preservation
    for (0..6) |i| {
        try std.testing.expect(reshaped.data[i] == i + 1);
    }
}

test "Reshape - With negative dimension" {
    tests_log.info("\n     test: Reshape - With negative dimension ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Reshape to 3x-1 (should become 3x2)
    var new_shape: [2]isize = [_]isize{ 3, @bitCast(@as(isize, -1)) };
    var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
    defer reshaped.deinit();

    // Verify shape
    try std.testing.expect(reshaped.shape[0] == 3);
    try std.testing.expect(reshaped.shape[1] == 2);
    try std.testing.expect(reshaped.size == 6);

    // Verify data preservation
    for (0..6) |i| {
        try std.testing.expect(reshaped.data[i] == i + 1);
    }
}

test "Reshape - With zero dimension" {
    tests_log.info("\n     test: Reshape - With zero dimension ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Reshape to 0x-1 (should keep first dimension as 2, and infer second as 3)
    var new_shape: [2]isize = [_]isize{ 0, @bitCast(@as(isize, -1)) };
    var reshaped = try TensMath.reshape(u8, &tensor, &new_shape, null);
    defer reshaped.deinit();

    // Verify shape
    try std.testing.expect(reshaped.shape[0] == 2);
    try std.testing.expect(reshaped.shape[1] == 3);
    try std.testing.expect(reshaped.size == 6);

    // Verify data preservation
    for (0..6) |i| {
        try std.testing.expect(reshaped.data[i] == i + 1);
    }
}

test "Reshape - Multiple negative dimensions (should fail)" {
    tests_log.info("\n     test: Reshape - Multiple negative dimensions ", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    // Try to reshape with multiple -1 dimensions (should fail)
    var invalid_shape: [2]isize = [_]isize{ @bitCast(@as(isize, -1)), @bitCast(@as(isize, -1)) };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.reshape(u8, &tensor, &invalid_shape, null));
}
