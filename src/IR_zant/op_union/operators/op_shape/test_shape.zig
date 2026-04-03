const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_shape);

test "get_split_output_shapes - default split" {
    tests_log.info("\n     test: get_split_output_shapes - default split", .{});

    // When split_sizes is null, should split into equal parts
    var input_shape = [_]usize{ 2, 3, 4 };
    const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, null, 1);
    defer {
        for (output_shapes) |shape| {
            pkgAllocator.allocator.free(shape);
        }
        pkgAllocator.allocator.free(output_shapes);
    }

    try std.testing.expectEqual(@as(usize, 1), output_shapes.len);
    try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
    try std.testing.expectEqual(@as(usize, 3), output_shapes[0][1]);
    try std.testing.expectEqual(@as(usize, 4), output_shapes[0][2]);
}

test "lean_shape_onnx basic operations2" {
    const allocator = std.testing.allocator;

    // Test 1: Basic operation
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with correct shape
        var output_shape = [_]usize{3};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output);

        try std.testing.expectEqual(@as(usize, 1), output.shape.len);
        try std.testing.expectEqual(@as(usize, 3), output.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[1]);
        try std.testing.expectEqual(@as(i64, 3), output.data[2]);
    }

    // Test 2: With start and end indices
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with correct shape
        var output_shape = [_]usize{2};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, 1, 3, &output);

        try std.testing.expectEqual(@as(usize, 1), output.shape.len);
        try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 3), output.data[1]);
    }

    // Test 3: Shape mismatch error
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with wrong shape
        var output_shape = [_]usize{2};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        // Should fail because output tensor shape doesn't match expected size (3)
        try std.testing.expectError(TensorError.ShapeMismatch, TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output));
    }
}

test "lean_shape_onnx operations and error cases" {
    const allocator = std.testing.allocator;

    // Test 1: Basic operation
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with correct shape
        var output_shape = [_]usize{3};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output);

        try std.testing.expectEqual(@as(usize, 1), output.shape.len);
        try std.testing.expectEqual(@as(usize, 3), output.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[1]);
        try std.testing.expectEqual(@as(i64, 3), output.data[2]);
    }

    // Test 2: With start and end indices
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with correct shape
        var output_shape = [_]usize{2};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, 1, 3, &output);

        try std.testing.expectEqual(@as(usize, 1), output.shape.len);
        try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 3), output.data[1]);
    }

    // Test 3: Shape mismatch error
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Create output tensor with wrong shape
        var output_shape = [_]usize{2};
        var output = try Tensor(i64).fromShape(&allocator, &output_shape);
        defer output.deinit();

        // Should fail because output tensor shape doesn't match expected size (3)
        try std.testing.expectError(TensorError.ShapeMismatch, TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output));
    }
}

test "shape_onnx basic operations" {
    const allocator = std.testing.allocator;

    // Test 1: Basic 3D tensor shape
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Get full shape
        var result = try TensMath.shape_onnx(f32, &tensor, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape.len);
        try std.testing.expectEqual(@as(usize, 3), result.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), result.data[0]);
        try std.testing.expectEqual(@as(i64, 2), result.data[1]);
        try std.testing.expectEqual(@as(i64, 3), result.data[2]);
    }

    // Test 2: With start and end indices
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Get shape[1:3]
        var result = try TensMath.shape_onnx(f32, &tensor, 1, 3);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), result.data[0]);
        try std.testing.expectEqual(@as(i64, 3), result.data[1]);
    }

    // Test 3: Negative indices
    {
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Get shape[-2:] (last 2 dimensions)
        var result = try TensMath.shape_onnx(f32, &tensor, -2, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(i64, 2), result.data[0]);
        try std.testing.expectEqual(@as(i64, 3), result.data[1]);
    }

    // Test 4: Edge cases
    {
        // Single dimension tensor
        var input_array = [_]f32{ 1, 2, 3, 4 };
        var shape = [_]usize{4};
        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        // Get full shape
        var result = try TensMath.shape_onnx(f32, &tensor, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape.len);
        try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
        try std.testing.expectEqual(@as(i64, 4), result.data[0]);

        // Out of bounds indices should be clamped
        var result2 = try TensMath.shape_onnx(f32, &tensor, -5, 5);
        defer result2.deinit();

        try std.testing.expectEqual(@as(usize, 1), result2.shape.len);
        try std.testing.expectEqual(@as(usize, 1), result2.shape[0]);
        try std.testing.expectEqual(@as(i64, 4), result2.data[0]);
    }
}

test "lean_shape_onnx basic operations" {
    tests_log.info("\n     test: lean_shape_onnx basic operations", .{});
    const allocator = pkgAllocator.allocator;

    // Test Case 1: Basic operation with pre-allocated output tensor
    {
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
        var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor.deinit();

        var outputArray = [_]i64{ 0, 0, 0 };
        var outputShape = [_]usize{3};
        var output = try Tensor(i64).fromArray(&allocator, &outputArray, &outputShape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output);

        // Check output data
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 2), output.data[1]);
        try std.testing.expectEqual(@as(i64, 3), output.data[2]);
    }

    // Test Case 2: Shape mismatch error
    {
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
        var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor.deinit();

        var outputArray = [_]i64{ 0, 0 };
        var outputShape = [_]usize{2};
        var output = try Tensor(i64).fromArray(&allocator, &outputArray, &outputShape);
        defer output.deinit();

        // Should return ShapeMismatch error
        try std.testing.expectError(TensorError.ShapeMismatch, TensMath.shape_onnx_lean(f32, i64, &tensor, null, null, &output));
    }

    // Test Case 3: With start and end parameters
    {
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
        var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
        defer tensor.deinit();

        var outputArray = [_]i64{ 0, 0 };
        var outputShape = [_]usize{2};
        var output = try Tensor(i64).fromArray(&allocator, &outputArray, &outputShape);
        defer output.deinit();

        try TensMath.shape_onnx_lean(f32, i64, &tensor, 1, 3, &output);

        // Check output data
        try std.testing.expectEqual(@as(i64, 2), output.data[0]);
        try std.testing.expectEqual(@as(i64, 3), output.data[1]);
    }
}

// test "get_shape_output_shape" {
//     const testing = std.testing;

//     // Test case 1: Basic shape without start/end
//     {
//         const input_shape = [_]usize{ 2, 3, 4 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 3), result[0]); // Should output [3] since no slicing
//     }

//     // Test case 2: Shape with start parameter
//     {
//         const input_shape = [_]usize{ 2, 3, 4, 5 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, 1, null);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 3), result[0]); // Should output [3] for dimensions 1 to end
//     }

//     // Test case 3: Shape with both start and end
//     {
//         const input_shape = [_]usize{ 2, 3, 4, 5, 6 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, 1, 3);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 2), result[0]); // Should output [2] for dimensions 1 to 3
//     }

//     // Test case 4: Shape with negative indices
//     {
//         const input_shape = [_]usize{ 2, 3, 4, 5 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, -2, -1);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 1), result[0]); // Should output [1] for dimensions -2 to -1
//     }

//     // Test case 5: Empty range
//     {
//         const input_shape = [_]usize{ 2, 3, 4 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, 2, 1);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] for invalid range
//     }

//     // Test case 6: Empty input shape
//     {
//         const input_shape = [_]usize{};
//         const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] for empty input
//     }

//     // Test case 7: Out of bounds indices
//     {
//         const input_shape = [_]usize{ 2, 3, 4 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, -5, 5);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 3), result[0]); // Should clamp indices and output [3]
//     }

//     // Test case 8: Single dimension input shape
//     {
//         const input_shape = [_]usize{42};
//         const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 1), result[0]); // Should output [1] for single dimension
//     }

//     // Test case 9: Start equals end
//     {
//         const input_shape = [_]usize{ 2, 3, 4 };
//         const result = try TensMath.get_shape_output_shape(&input_shape, 1, 1);
//         defer pkgAllocator.allocator.free(result);
//         try testing.expectEqual(@as(usize, 1), result.len);
//         try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] when start equals end
//     }
// }
