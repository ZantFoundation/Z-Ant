const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_shape);

test "Empty tensor list error" {
    tests_log.info("\n     test: Empty tensor list error", .{});
    const empty_shapes: []const []const usize = &[_][]const usize{};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.get_concatenate_output_shape(empty_shapes, 0));
}

test "get_concatenate_output_shape" {
    tests_log.info("\n     test: get_concatenate_output_shape \n", .{});

    const allocator = pkgAllocator.allocator;

    // Test concatenation along axis 0
    var shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 3, 3 },
        &[_]usize{ 1, 3 },
    };

    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes, 0);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 6), output_shape[0]); // 2 + 3 + 1
        try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // unchanged
    }

    // Test concatenation along axis 1
    const shapes_axis1 = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 2, 2 },
    };

    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes_axis1, 1);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // unchanged
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 3 + 2
    }

    // Test concatenation along negative axis
    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes_axis1, -1);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // unchanged
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 3 + 2
    }

    // Test error cases
    var empty_shapes = [_][]const usize{};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.get_concatenate_output_shape(&empty_shapes, 0));

    var mismatched_shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 3, 4 }, // different non-concat dimension
    };
    try std.testing.expectError(TensorError.MismatchedShape, TensMath.get_concatenate_output_shape(&mismatched_shapes, 0));
}

test "Concatenate tensors along axis 0" {
    tests_log.info("\n     test: Concatenate tensors along axis 0", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 0);
    defer result_tensor.deinit();

    const expected_data: [4][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    try std.testing.expect(result_tensor.shape[0] == 4);
    try std.testing.expect(result_tensor.shape[1] == 2);

    for (0..4) |i| {
        for (0..2) |j| {
            //tests_log.debug("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 2 + j] });
            try std.testing.expect(result_tensor.data[i * 2 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along axis 1" {
    tests_log.info("\n     test: Concatenate tensors along axis 1", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 1);
    defer result_tensor.deinit();

    const expected_data: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 5.0, 6.0 },
        [_]f32{ 3.0, 4.0, 7.0, 8.0 },
    };
    result_tensor.print();

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 4);

    for (0..2) |i| {
        for (0..4) |j| {
            //tests_log.info("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 4 + j] });
            try std.testing.expect(result_tensor.data[i * 4 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along negative axis" {
    tests_log.info("\n     test: Concatenate tensors along negative axis", .{});
    var allocator = std.testing.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    // Initialize tensors from arrays
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    // Perform concatenation along axis -1 (equivalent to axis 1)
    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, -1);
    defer result_tensor.deinit();

    const expected_data: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 5.0, 6.0 },
        [_]f32{ 3.0, 4.0, 7.0, 8.0 },
    };

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 4);

    for (0..2) |i| {
        for (0..4) |j| {
            try std.testing.expect(result_tensor.data[i * 4 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate 3D tensors along axis 2" {
    tests_log.info("\n     test: Concatenate 3D tensors along axis 2", .{});
    var allocator = std.testing.allocator;

    // Tensor A: shape [2, 2, 2]
    var inputArrayA: [2][2][2]f32 = [_][2][2]f32{
        [_][2]f32{ [_]f32{ 1.0, 2.0 }, [_]f32{ 3.0, 4.0 } },
        [_][2]f32{ [_]f32{ 5.0, 6.0 }, [_]f32{ 7.0, 8.0 } },
    };
    var shapeA: [3]usize = [_]usize{ 2, 2, 2 };
    var tA = try Tensor(f32).fromArray(&allocator, &inputArrayA, &shapeA);
    defer tA.deinit();

    // Tensor B: shape [2, 2, 3]
    var inputArrayB: [2][2][3]f32 = [_][2][3]f32{
        [_][3]f32{ [_]f32{ 9.0, 10.0, 11.0 }, [_]f32{ 12.0, 13.0, 14.0 } },
        [_][3]f32{ [_]f32{ 15.0, 16.0, 17.0 }, [_]f32{ 18.0, 19.0, 20.0 } },
    };
    var shapeB: [3]usize = [_]usize{ 2, 2, 3 };
    var tB = try Tensor(f32).fromArray(&allocator, &inputArrayB, &shapeB);
    defer tB.deinit();

    var tensors = [_]Tensor(f32){ tA, tB };

    // Perform concatenation along axis 2
    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 2);
    defer result_tensor.deinit();

    const expected_data: [2][2][5]f32 = [_][2][5]f32{
        [_][5]f32{
            [_]f32{ 1.0, 2.0, 9.0, 10.0, 11.0 },
            [_]f32{ 3.0, 4.0, 12.0, 13.0, 14.0 },
        },
        [_][5]f32{
            [_]f32{ 5.0, 6.0, 15.0, 16.0, 17.0 },
            [_]f32{ 7.0, 8.0, 18.0, 19.0, 20.0 },
        },
    };

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 2);
    try std.testing.expect(result_tensor.shape[2] == 5);

    for (0..2) |i| {
        for (0..2) |j| {
            for (0..5) |k| {
                try std.testing.expect(result_tensor.data[i * 2 * 5 + j * 5 + k] == expected_data[i][j][k]);
            }
        }
    }
}

test "Concatenate tensors with mismatched shapes" {
    tests_log.info("\n     test: Concatenate tensors with mismatched shapes", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][3]f32 = [_][3]f32{
        [_]f32{ 5.0, 6.0, 7.0 },
        [_]f32{ 8.0, 9.0, 10.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 };
    var shape2: [2]usize = [_]usize{ 2, 3 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    // Should fail when trying to concatenate along axis 0 due to mismatched shapes
    try std.testing.expectError(TensorError.MismatchedShape, TensMath.concatenate(f32, &allocator, &tensors, 0));

    // Should succeed when concatenating along axis 1
    var result = try TensMath.concatenate(f32, &allocator, &tensors, 1);
    defer result.deinit();

    try std.testing.expect(result.shape[0] == 2);
    try std.testing.expect(result.shape[1] == 5);

    const expected_data: [2][5]f32 = [_][5]f32{
        [_]f32{ 1.0, 2.0, 5.0, 6.0, 7.0 },
        [_]f32{ 3.0, 4.0, 8.0, 9.0, 10.0 },
    };

    for (0..2) |i| {
        for (0..5) |j| {
            try std.testing.expect(result.data[i * 5 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors with invalid axis" {
    tests_log.info("\n     test: Concatenate tensors with invalid axis", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    // Should fail when axis is out of bounds
    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.concatenate(f32, &allocator, &tensors, 2));
    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.concatenate(f32, &allocator, &tensors, -3));
}

test "get_concatenate_output_shape - 3D tensors" {
    tests_log.info("\n     test: get_concatenate_output_shape - 3D tensors", .{});
    const allocator = pkgAllocator.allocator;

    // Test shapes for 3D tensors
    var shapes = [_][]const usize{
        &[_]usize{ 2, 2, 2 },
        &[_]usize{ 2, 2, 3 },
    };

    // Test concatenation along last axis (axis 2)
    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes, 2);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 5), output_shape[2]); // 2 + 3
    }

    // Test concatenation along middle axis (axis 1)
    var shapes_axis1 = [_][]const usize{
        &[_]usize{ 2, 2, 3 },
        &[_]usize{ 2, 3, 3 },
    };

    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes_axis1, 1);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 2 + 3
        try std.testing.expectEqual(@as(usize, 3), output_shape[2]);
    }
}

test "get_concatenate_output_shape - mismatched shapes" {
    tests_log.info("\n     test: get_concatenate_output_shape - mismatched shapes", .{});

    // Test shapes with mismatched dimensions
    var shapes = [_][]const usize{
        &[_]usize{ 2, 2 },
        &[_]usize{ 2, 3 },
    };

    // Should fail along axis 0 (mismatched non-concat dimensions)
    try std.testing.expectError(TensorError.MismatchedShape, TensMath.get_concatenate_output_shape(&shapes, 0));

    // Should succeed along axis 1
    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes, 1);
        defer pkgAllocator.allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 2 + 3
    }
}

//Mismatched rank now supported

test "get_concatenate_output_shape - invalid axis" {
    tests_log.info("\n     test: get_concatenate_output_shape - invalid axis", .{});

    var shapes = [_][]const usize{
        &[_]usize{ 2, 2 },
        &[_]usize{ 2, 2 },
    };

    // Test positive out of bounds axis
    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.get_concatenate_output_shape(&shapes, 2));

    // Test negative out of bounds axis
    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.get_concatenate_output_shape(&shapes, -3));

    // Test valid negative axis
    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes, -1);
        defer pkgAllocator.allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]); // 2 + 2
    }
}
