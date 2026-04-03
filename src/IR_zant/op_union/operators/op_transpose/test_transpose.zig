const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "transpose" {
    tests_log.info("\n     test: transpose ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensor_transposed = try TensMath.transpose2D(u8, &tensor);
    defer tensor_transposed.deinit();

    try std.testing.expect(tensor_transposed.data[0] == 1);
    try std.testing.expect(tensor_transposed.data[1] == 4);
    try std.testing.expect(tensor_transposed.data[2] == 2);
    try std.testing.expect(tensor_transposed.data[3] == 5);
    try std.testing.expect(tensor_transposed.data[4] == 3);
    try std.testing.expect(tensor_transposed.data[5] == 6);
}

test "transpose multi-dimensions default" {
    tests_log.info("\n     test: transpose multi-dimensions ", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input Array and shape
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

    var tensor_transposed = try TensMath.transposeDefault(u8, &tensor);
    defer tensor_transposed.deinit();

    for (0..tensor.size) |i| {
        try std.testing.expect(tensor_transposed.data[i] == tensor.data[i]);
    }
}

test "transpose_onnx basic operations" {
    tests_log.info("\n     test: transpose_onnx basic operations", .{});
    const allocator = pkgAllocator.allocator;

    // Test Case 1: Basic 2D transpose without perm
    {
        // Create a 2x3 tensor
        var inputArray = [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
        };
        var shape = [_]usize{ 2, 3 };
        var tensor1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor1.deinit();

        var outputArray = [_][2]f32{
            [_]f32{ 0.0, 0.0 },
            [_]f32{ 0.0, 0.0 },
            [_]f32{ 0.0, 0.0 },
        };
        var outputShape = [_]usize{ 3, 2 };
        var output1 = try Tensor(f32).fromArray(&allocator, &outputArray, &outputShape);
        defer output1.deinit();

        // Transpose without perm (should reverse dimensions)
        try TensMath.transpose_onnx_lean(f32, &tensor1, null, &output1, pkgAllocator.allocator);

        // Check shape
        try std.testing.expectEqual(@as(usize, 3), output1.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output1.shape[1]);

        // Expected data after transpose: [1, 4, 2, 5, 3, 6]
        const expected = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
        for (output1.data, 0..) |val, i| {
            try std.testing.expectEqual(expected[i], val);
        }
    }

    // Test Case 2: 3D tensor with custom permutation
    {
        // Create a 2x2x3 tensor
        var inputArray = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
            },
            [_][3]f32{
                [_]f32{ 7.0, 8.0, 9.0 },
                [_]f32{ 10.0, 11.0, 12.0 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor2.deinit();

        var outputArray = [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        };
        var outputShape = [_]usize{ 3, 2, 2 };
        var output2 = try Tensor(f32).fromArray(&allocator, &outputArray, &outputShape);
        defer output2.deinit();

        // Transpose with perm [2, 1, 0]
        const perm = [_]usize{ 2, 1, 0 };
        try TensMath.transpose_onnx_lean(f32, &tensor2, &perm, &output2, pkgAllocator.allocator);

        // Check shape
        try std.testing.expectEqual(@as(usize, 3), output2.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output2.shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output2.shape[2]);

        // Expected data after transpose with perm [2, 1, 0]
        const expected = [_]f32{ 1.0, 7.0, 4.0, 10.0, 2.0, 8.0, 5.0, 11.0, 3.0, 9.0, 6.0, 12.0 };
        for (output2.data, 0..) |val, i| {
            try std.testing.expectEqual(expected[i], val);
        }
    }

    // Test Case 3: Error handling - invalid permutation
    {
        var inputArray = [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
        };
        var shape = [_]usize{ 2, 3 };
        var tensor3 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor3.deinit();

        var outputArray = [_][2]f32{
            [_]f32{ 0.0, 0.0 },
            [_]f32{ 0.0, 0.0 },
            [_]f32{ 0.0, 0.0 },
        };
        var outputShape = [_]usize{ 3, 2 };
        var output3 = try Tensor(f32).fromArray(&allocator, &outputArray, &outputShape);
        defer output3.deinit();
    }
}

test "get_transpose_output_shape basic operations" {
    tests_log.info("\n     test: get_transpose_output_shape basic operations", .{});
    const allocator = pkgAllocator.allocator;

    // Test Case 1: Basic 2D shape without perm
    {
        const input_shape = [_]usize{ 2, 3 };
        const output_shape = try TensMath.get_transpose_output_shape(&input_shape, null);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 3), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[1]);
    }

    // Test Case 2: 3D shape with custom permutation
    {
        const input_shape = [_]usize{ 2, 3, 4 };
        const perm = [_]usize{ 2, 0, 1 };
        const output_shape = try TensMath.get_transpose_output_shape(&input_shape, &perm);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 3), output_shape[2]);
    }
}
