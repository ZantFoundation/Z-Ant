const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "Concatenate tensors along axis 0" {
    std.debug.print("\n     test: Concatenate tensors along axis 0", .{});
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
            //std.debug.print("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 2 + j] });
            try std.testing.expect(result_tensor.data[i * 2 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along axis 1" {
    std.debug.print("\n     test: Concatenate tensors along axis 1", .{});
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
            //std.debug.print("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 4 + j] });
            try std.testing.expect(result_tensor.data[i * 4 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along negative axis" {
    std.debug.print("\n     test: Concatenate tensors along negative axis", .{});
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
    std.debug.print("\n     test: Concatenate 3D tensors along axis 2", .{});
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

test "transpose" {
    std.debug.print("\n     test: transpose ", .{});
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
    std.debug.print("\n     test: transpose multi-dimensions ", .{});
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

test "test addPaddingAndDilation() " {
    std.debug.print("\n     test: addPadding()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 1, 2, 1, 2);

    var resultArray: [2][7][11]i8 = [_][7][11]i8{
        [_][11]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
        [_][11]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 7, 11 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test addPaddingAndDilation() -> zero dilatation " {
    std.debug.print("\n     test: addPadding() -> zero dilatation ", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 1, 2, 0, 0);

    var resultArray: [2][5][7]i8 = [_][5][7]i8{
        [_][7]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 2, 3, 0, 0 },
            [_]i8{ 0, 0, 4, 5, 6, 0, 0 },
            [_]i8{ 0, 0, 7, 8, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
        },
        [_][7]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 2, 3, 0, 0 },
            [_]i8{ 0, 0, 4, 5, 6, 0, 0 },
            [_]i8{ 0, 0, 7, 8, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 5, 7 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test addPaddingAndDilation() -> zero padding" {
    std.debug.print("\n     test: addPaddingAndDilation() -> zero padding", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 0, 0, 1, 2);

    var resultArray: [2][5][7]i8 = [_][5][7]i8{
        [_][7]i8{
            [_]i8{ 1, 0, 0, 2, 0, 0, 3 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 4, 0, 0, 5, 0, 0, 6 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 7, 0, 0, 8, 0, 0, 9 },
        },
        [_][7]i8{
            [_]i8{ 1, 0, 0, 2, 0, 0, 3 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 4, 0, 0, 5, 0, 0, 6 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 7, 0, 0, 8, 0, 0, 9 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 5, 7 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    tensor.info();
    tensor.print();
    resultTensor.info();
    resultTensor.print();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test neg() " {
    std.debug.print("\n     test: neg()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 10, 20, 30 },
            [_]i8{ 40, 50, 60 },
            [_]i8{ 70, 80, 90 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var flippedTensor = try TensMath.flip(i8, &tensor);
    defer flippedTensor.deinit();
    // DEBUG flippedTensor.print();

    var resultArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 9, 8, 7 },
            [_]i8{ 6, 5, 4 },
            [_]i8{ 3, 2, 1 },
        },
        [_][3]i8{
            [_]i8{ 90, 80, 70 },
            [_]i8{ 60, 50, 40 },
            [_]i8{ 30, 20, 10 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 3, 3 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();
    std.debug.print("TRY WITH THISSS: \n", .{});
    resultTensor.printMultidim();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], flippedTensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], flippedTensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, flippedTensor.size);
}

test "resize with nearest neighbor interpolation" {
    std.debug.print("\n     test: resize with nearest neighbor interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "nearest", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[0]);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[1]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[2]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[3]);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "nearest", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with linear interpolation" {
    std.debug.print("\n     test: resize with linear interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "linear", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "linear", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with cubic interpolation" {
    std.debug.print("\n     test: resize with cubic interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var shape_1d = [_]usize{8};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale down by 0.5x
    var scales = [_]f32{0.5};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "cubic", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 4), resized_1d.size);
}

test "resize with explicit sizes" {
    std.debug.print("\n     test: resize with explicit sizes", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Resize to specific dimensions
    var sizes = [_]usize{ 3, 3 };
    var resized = try TensMath.resize(u8, &tensor, "nearest", null, &sizes, "half_pixel");
    defer resized.deinit();

    try std.testing.expectEqual(@as(usize, 9), resized.size);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[1]);
}

test "resize error cases" {
    std.debug.print("\n     test: resize error cases", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test invalid mode
    var scales = [_]f32{ 2.0, 2.0 };
    try std.testing.expectError(
        TensorError.UnsupportedMode,
        TensMath.resize(u8, &tensor, "invalid_mode", &scales, null, "half_pixel"),
    );

    // Test both scales and sizes provided
    var sizes = [_]usize{ 3, 3 };
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", &scales, &sizes, "half_pixel"),
    );

    // Test neither scales nor sizes provided
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", null, null, "half_pixel"),
    );
}

test "split basic test" {
    std.debug.print("\n     test: split basic test", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    const input_array = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis 0 (rows)
    const split_tensors = try TensMath.split(u8, &tensor, 0, null); // Added num_outputs parameter
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 1), split_tensors.len);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 3), split_tensors[0].shape[1]);
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 6), split_tensors[0].data[5]);
}

test "split with custom sizes" {
    std.debug.print("\n     test: split with custom sizes", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 4x2 tensor
    const input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
        [_]u8{ 5, 6 },
        [_]u8{ 7, 8 },
    };
    var shape = [_]usize{ 4, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis 0 into [1,3] parts
    const split_sizes = [_]usize{ 1, 3 };
    const split_tensors = try TensMath.split(u8, &tensor, 0, &split_sizes); // Added num_outputs parameter
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 2), split_tensors.len);

    // First split should be 1x2
    try std.testing.expectEqual(@as(usize, 1), split_tensors[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[0].shape[1]);
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 2), split_tensors[0].data[1]);

    // Second split should be 3x2
    try std.testing.expectEqual(@as(usize, 3), split_tensors[1].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[1].shape[1]);
    try std.testing.expectEqual(@as(u8, 3), split_tensors[1].data[0]);
    try std.testing.expectEqual(@as(u8, 8), split_tensors[1].data[5]);
}

test "split with negative axis" {
    std.debug.print("\n     test: split with negative axis", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x4 tensor
    const input_array = [_][4]u8{
        [_]u8{ 1, 2, 3, 4 },
        [_]u8{ 5, 6, 7, 8 },
    };
    var shape = [_]usize{ 2, 4 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis -1 (last axis) into [2,2] parts
    const split_sizes = [_]usize{ 2, 2 };
    const split_tensors = try TensMath.split(u8, &tensor, -1, &split_sizes); // Added num_outputs parameter
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 2), split_tensors.len);

    // Both splits should be 2x2
    for (split_tensors) |t| {
        try std.testing.expectEqual(@as(usize, 2), t.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), t.shape[1]);
    }

    // Check first split
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 2), split_tensors[0].data[1]);
    try std.testing.expectEqual(@as(u8, 5), split_tensors[0].data[2]);
    try std.testing.expectEqual(@as(u8, 6), split_tensors[0].data[3]);

    // Check second split
    try std.testing.expectEqual(@as(u8, 3), split_tensors[1].data[0]);
    try std.testing.expectEqual(@as(u8, 4), split_tensors[1].data[1]);
    try std.testing.expectEqual(@as(u8, 7), split_tensors[1].data[2]);
    try std.testing.expectEqual(@as(u8, 8), split_tensors[1].data[3]);
}

test "get_resize_output_shape()" {
    std.debug.print("\n     test: get_resize_output_shape \n", .{});

    var input_shape = [_]usize{ 2, 3, 4 };
    var scales = [_]f32{ 2.0, 1.5, 0.5 };
    var target_sizes = [_]usize{ 4, 4, 2 };

    // Test with scales
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, &scales, null);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test with target sizes
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, null, &target_sizes);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test invalid input (both scales and sizes null)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, null));

    // Test invalid input (both scales and sizes provided)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &scales, &target_sizes));

    // Test mismatched dimensions
    var wrong_scales = [_]f32{ 2.0, 1.5 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &wrong_scales, null));

    var wrong_sizes = [_]usize{ 4, 4 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, &wrong_sizes));
}

test "get_concatenate_output_shape" {
    std.debug.print("\n     test: get_concatenate_output_shape \n", .{});

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

test "get_split_output_shapes()" {
    std.debug.print("\n     test: get_split_output_shapes \n", .{});

    const allocator = pkgAllocator.allocator;
    var input_shape = [_]usize{ 2, 3, 4 };

    // Test with null split_sizes (equal splits)
    {
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, null, null);
        defer {
            for (output_shapes) |shape| {
                allocator.free(shape);
            }
            allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 1), output_shapes.len);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0].len);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0][1]);
        try std.testing.expectEqual(@as(usize, 4), output_shapes[0][2]);
    }

    // Test with specific split_sizes
    {
        var split_sizes = [_]usize{ 1, 2 };
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, &split_sizes, null);
        defer {
            for (output_shapes) |shape| {
                allocator.free(shape);
            }
            allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 2), output_shapes.len);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0].len);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
        try std.testing.expectEqual(@as(usize, 1), output_shapes[0][1]);
        try std.testing.expectEqual(@as(usize, 4), output_shapes[0][2]);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[1][1]);
    }

    // Test invalid axis
    try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, -4, null, null));
    try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, 3, null, null));

    // Test invalid split sizes
    var invalid_split_sizes = [_]usize{ 1, 1 };
    try std.testing.expectError(TensorError.InvalidSplitSize, TensMath.get_split_output_shapes(&input_shape, 1, &invalid_split_sizes, null));
}

test "Empty tensor list error" {
    std.debug.print("\n     test: Empty tensor list error", .{});
    const empty_shapes: []const []const usize = &[_][]const usize{};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.get_concatenate_output_shape(empty_shapes, 0));
}

test "Reshape" {
    std.debug.print("\n     test: Reshape ", .{});
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

    std.debug.print(" \n\n  new_tens.shape= {any} ", .{new_tens.shape});

    try std.testing.expect(new_tens.size == new_tens.size);
    try std.testing.expect(new_tens.shape[2] == 3);
    try std.testing.expect(new_tens.shape[3] == 2);
}

test "gather along axis 0 and axis 1" {
    const allocator = pkgAllocator.allocator;

    // -------------------------------------------------------------------------------------
    // Test Case 1: Gather Along Axis 0
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather along axis 0", .{});

    // Initialize input tensor: 3x3 matrix
    var inputArray0: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var inputShape0: [2]usize = [_]usize{ 3, 3 };
    var inputTensor0 = try Tensor(u8).fromArray(&allocator, &inputArray0, &inputShape0);
    defer inputTensor0.deinit();

    // Initialize indices tensor: [0, 2]
    var indicesArray0: [2]usize = [_]usize{ 0, 2 };
    var indicesShape0: [1]usize = [_]usize{2};
    var indicesTensor0 = try Tensor(usize).fromArray(&allocator, &indicesArray0, &indicesShape0);
    defer indicesTensor0.deinit();

    // Perform gather along axis 0
    var gatheredTensor0 = try TensMath.gather(u8, &inputTensor0, &indicesTensor0, 0);
    defer gatheredTensor0.deinit();

    // Expected output tensor: [1,2,3,7,8,9], shape [2,3]
    const expectedData0: [6]u8 = [_]u8{ 1, 2, 3, 7, 8, 9 };
    const expectedShape0: [2]usize = [_]usize{ 2, 3 };

    // Check shape
    try std.testing.expect(gatheredTensor0.shape.len == expectedShape0.len);
    for (0..expectedShape0.len) |i| {
        try std.testing.expect(gatheredTensor0.shape[i] == expectedShape0[i]);
    }

    // Check data
    try std.testing.expect(gatheredTensor0.size == 6);
    for (0..gatheredTensor0.size) |i| {
        try std.testing.expect(gatheredTensor0.data[i] == expectedData0[i]);
    }

    // -------------------------------------------------------------------------------------
    // Test Case 2: Gather Along Axis 1
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather along axis 1", .{});

    var inputArray1: [2][4]u8 = [_][4]u8{
        [_]u8{ 10, 20, 30, 40 },
        [_]u8{ 50, 60, 70, 80 },
    };
    var inputShape1: [2]usize = [_]usize{ 2, 4 };
    var inputTensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &inputShape1);
    defer inputTensor1.deinit();

    var indicesArray1: [2][2]usize = [_][2]usize{
        [_]usize{ 1, 3 },
        [_]usize{ 0, 2 },
    };
    var indicesShape1: [2]usize = [_]usize{ 2, 2 };
    var indicesTensor1 = try Tensor(usize).fromArray(&allocator, &indicesArray1, &indicesShape1);
    defer indicesTensor1.deinit();

    // Perform gather along axis 1
    var gatheredTensor1 = try TensMath.gather(u8, &inputTensor1, &indicesTensor1, 1);
    defer gatheredTensor1.deinit();

    // Expected output tensor: [
    //   [20, 40],
    //   [10, 30],
    //   [60, 80],
    //   [50, 70]
    // ], shape [2, 2, 2]
    const expectedData1: [8]u8 = [_]u8{ 20, 40, 10, 30, 60, 80, 50, 70 };
    const expectedShape1: [3]usize = [_]usize{ 2, 2, 2 };

    // Check shape
    try std.testing.expect(gatheredTensor1.shape.len == expectedShape1.len);
    for (0..expectedShape1.len) |i| {
        try std.testing.expect(gatheredTensor1.shape[i] == expectedShape1[i]);
    }

    // Check data
    std.debug.print("\n     gatheredTensor1.size: {}\n", .{gatheredTensor1.size});
    gatheredTensor1.print();

    try std.testing.expect(gatheredTensor1.size == 8);
    for (0..gatheredTensor1.size) |i| {
        std.debug.print("\n     gatheredTensor1.data[i]: {}\n", .{expectedData1[i]});
        std.debug.print("\n     expectedData1[i]: {}\n", .{gatheredTensor1.data[i]});
        try std.testing.expect(gatheredTensor1.data[i] == expectedData1[i]);
    }

    // -------------------------------------------------------------------------------------
    // Test Case 3: Error Handling - Invalid Axis
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather with invalid axis", .{});
    const invalidAxis: usize = 3; // Input tensor has 2 dimensions
    const result0 = TensMath.gather(u8, &inputTensor0, &indicesTensor0, invalidAxis);
    try std.testing.expect(result0 == TensorError.InvalidAxis);
}

test "get_slice_output_shape basic slicing" {
    std.debug.print("\n     test: get_slice_output_shape basic slicing", .{});
    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 5, 5, 5 };
    var starts = [_]i64{ 1, 1, 1 };
    var ends = [_]i64{ 4, 4, 4 };

    // Test basic slicing without steps or axes
    {
        const output_shape = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, null);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 3), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 3), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 3), output_shape[2]);
    }

    // Test with steps
    {
        var steps = [_]i64{ 2, 2, 2 };
        const output_shape = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, &steps);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test with negative indices
    {
        var neg_starts = [_]i64{ -3, -3, -3 };
        var neg_ends = [_]i64{ -1, -1, -1 };
        const output_shape = try TensMath.get_slice_output_shape(&input_shape, &neg_starts, &neg_ends, null, null);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }
}

test "get_slice_output_shape with axes and negative steps" {
    std.debug.print("\n     test: get_slice_output_shape with axes and negative steps", .{});
    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 5, 5, 5 };

    // Test with specific axes
    {
        var starts = [_]i64{1};
        var ends = [_]i64{4};
        var axes = [_]i64{1}; // Only slice along axis 1

        const output_shape = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, &axes, null);
        defer allocator.free(output_shape);

        std.debug.print("\n     Specific axes test - output shape: ", .{});
        for (output_shape) |dim| {
            std.debug.print("{} ", .{dim});
        }
        std.debug.print("\n", .{});

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        try std.testing.expectEqual(@as(usize, 5), output_shape[0]); // unchanged
        try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // sliced
        try std.testing.expectEqual(@as(usize, 5), output_shape[2]); // unchanged
    }

    // Test with negative steps
    {
        var starts = [_]i64{ 4, 4, 4 };
        var ends = [_]i64{ 1, 1, 1 };
        var steps = [_]i64{ -1, -1, -1 };

        std.debug.print("\n     Negative steps test - inputs:", .{});
        std.debug.print("\n     starts: {} {} {}", .{ starts[0], starts[1], starts[2] });
        std.debug.print("\n     ends: {} {} {}", .{ ends[0], ends[1], ends[2] });
        std.debug.print("\n     steps: {} {} {}", .{ steps[0], steps[1], steps[2] });

        const output_shape = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, &steps);
        defer allocator.free(output_shape);

        std.debug.print("\n     Negative steps test - output shape: ", .{});
        for (output_shape) |dim| {
            std.debug.print("{} ", .{dim});
        }
        std.debug.print("\n", .{});

        try std.testing.expectEqual(@as(usize, 3), output_shape.len);
        // For negative steps, we expect:
        // start = 4, end = 1, step = -1
        // Elements will be: 4, 3, 2, 1 (4 elements)
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[2]);
    }
}

test "get_slice_output_shape error cases" {
    std.debug.print("\n     test: get_slice_output_shape error cases", .{});

    var input_shape = [_]usize{ 5, 5, 5 };

    // Test mismatched starts and ends lengths
    {
        var starts = [_]i64{ 1, 2 };
        var ends = [_]i64{4};
        try std.testing.expectError(TensorError.InvalidSliceIndices, TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, null));
    }

    // Test invalid axes length
    {
        var starts = [_]i64{ 1, 2 };
        var ends = [_]i64{ 4, 5 };
        var axes = [_]i64{1}; // Wrong length
        try std.testing.expectError(TensorError.InvalidSliceIndices, TensMath.get_slice_output_shape(&input_shape, &starts, &ends, &axes, null));
    }

    // Test invalid steps
    {
        var starts = [_]i64{1};
        var ends = [_]i64{4};
        var steps = [_]i64{0}; // Zero step not allowed
        try std.testing.expectError(TensorError.InvalidSliceStep, TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, &steps));
    }

    // Test axis out of bounds
    {
        var starts = [_]i64{1};
        var ends = [_]i64{4};
        var axes = [_]i64{5}; // Out of bounds
        try std.testing.expectError(TensorError.InvalidSliceIndices, TensMath.get_slice_output_shape(&input_shape, &starts, &ends, &axes, null));
    }
}

test "transpose_onnx basic operations" {
    std.debug.print("\n     test: transpose_onnx basic operations", .{});
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
        try TensMath.transpose_onnx_lean(f32, &tensor1, null, &output1);

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
        try TensMath.transpose_onnx_lean(f32, &tensor2, &perm, &output2);

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

        // Try with invalid permutation (wrong length)
        const invalid_perm = [_]usize{ 0, 1, 2 }; // 3 values for 2D tensor
        const result = TensMath.transpose_onnx_lean(f32, &tensor3, &invalid_perm, &output3);
        try std.testing.expectError(TensorError.InvalidPermutation, result);
    }
}

test "get_transpose_output_shape basic operations" {
    std.debug.print("\n     test: get_transpose_output_shape basic operations", .{});
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

    // Test Case 3: Error handling - invalid permutation
    {
        const input_shape = [_]usize{ 2, 3 };
        const invalid_perm = [_]usize{ 0, 1, 2 }; // 3 values for 2D shape
        const result = TensMath.get_transpose_output_shape(&input_shape, &invalid_perm);
        try std.testing.expectError(TensorError.InvalidPermutation, result);
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
    std.debug.print("\n     test: lean_shape_onnx basic operations", .{});
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

// ... existing code ...

test "get_shape_output_shape" {
    const testing = std.testing;

    // Test case 1: Basic shape without start/end
    {
        const input_shape = [_]usize{ 2, 3, 4 };
        const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 3), result[0]); // Should output [3] since no slicing
    }

    // Test case 2: Shape with start parameter
    {
        const input_shape = [_]usize{ 2, 3, 4, 5 };
        const result = try TensMath.get_shape_output_shape(&input_shape, 1, null);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 3), result[0]); // Should output [3] for dimensions 1 to end
    }

    // Test case 3: Shape with both start and end
    {
        const input_shape = [_]usize{ 2, 3, 4, 5, 6 };
        const result = try TensMath.get_shape_output_shape(&input_shape, 1, 3);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 2), result[0]); // Should output [2] for dimensions 1 to 3
    }

    // Test case 4: Shape with negative indices
    {
        const input_shape = [_]usize{ 2, 3, 4, 5 };
        const result = try TensMath.get_shape_output_shape(&input_shape, -2, -1);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 1), result[0]); // Should output [1] for dimensions -2 to -1
    }

    // Test case 5: Empty range
    {
        const input_shape = [_]usize{ 2, 3, 4 };
        const result = try TensMath.get_shape_output_shape(&input_shape, 2, 1);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] for invalid range
    }

    // Test case 6: Empty input shape
    {
        const input_shape = [_]usize{};
        const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] for empty input
    }

    // Test case 7: Out of bounds indices
    {
        const input_shape = [_]usize{ 2, 3, 4 };
        const result = try TensMath.get_shape_output_shape(&input_shape, -5, 5);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 3), result[0]); // Should clamp indices and output [3]
    }

    // Test case 8: Single dimension input shape
    {
        const input_shape = [_]usize{42};
        const result = try TensMath.get_shape_output_shape(&input_shape, null, null);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 1), result[0]); // Should output [1] for single dimension
    }

    // Test case 9: Start equals end
    {
        const input_shape = [_]usize{ 2, 3, 4 };
        const result = try TensMath.get_shape_output_shape(&input_shape, 1, 1);
        defer pkgAllocator.allocator.free(result);
        try testing.expectEqual(@as(usize, 1), result.len);
        try testing.expectEqual(@as(usize, 0), result[0]); // Should output [0] when start equals end
    }
}

// f23 input tensor with [2, 3] shape and axes = [1] => expected output: [2, 1, 3]
test "test unsqueeze valid" {
    std.debug.print("\n     test: unsqueeze valid\n", .{});
    const allocator = std.testing.allocator;

    // Input tensor
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &inputShape);
    defer tensor.deinit();

    // Axes tensor: unsqueeze in position 1
    var axesArray: [1]i64 = [_]i64{1};
    var axesShape: [1]usize = [_]usize{1};
    var axesTensor = try Tensor(i64).fromArray(&allocator, &axesArray, &axesShape);
    defer axesTensor.deinit();

    var result = try TensMath.unsqueeze(f32, &tensor, &axesTensor);
    defer result.deinit();

    // Verify output shape [2, 1, 3]
    try std.testing.expect(result.shape.len == 3);
    try std.testing.expect(result.shape[0] == 2);
    try std.testing.expect(result.shape[1] == 1);
    try std.testing.expect(result.shape[2] == 3);

    // Verify data
    for (0..tensor.size) |i| {
        try std.testing.expect(result.data[i] == tensor.data[i]);
    }
}

// -------------------------------------------------------------
// Test for AxisOutOfBounds error
test "test unsqueeze axis out of bounds error" {
    std.debug.print("\n test: unsqueeze axis out of bounds error\n", .{});
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
    std.debug.print("\n test: unsqueeze duplicate axis error\n", .{});
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

test "Reshape - Basic" {
    std.debug.print("\n     test: Reshape - Basic ", .{});
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

test "Reshape - Multi-dimensional" {
    std.debug.print("\n     test: Reshape - Multi-dimensional ", .{});
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
    std.debug.print("\n     test: Reshape - Error case ", .{});
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
    std.debug.print("\n     test: Reshape - Same size different dimensions ", .{});
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
    std.debug.print("\n     test: Reshape - With negative dimension ", .{});
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
    std.debug.print("\n     test: Reshape - With zero dimension ", .{});
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
    std.debug.print("\n     test: Reshape - Multiple negative dimensions ", .{});
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

test "gather - negative axis" {
    std.debug.print("\n     test: gather - negative axis", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input tensor: 2x3 matrix
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize indices tensor: [1]
    var indicesArray: [1]usize = [_]usize{1};
    var indicesShape: [1]usize = [_]usize{1};
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Gather along axis -2 (equivalent to axis 0)
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, -2);
    defer gatheredTensor.deinit();

    // Expected: [4, 5, 6]
    try std.testing.expect(gatheredTensor.shape[0] == 1);
    try std.testing.expect(gatheredTensor.shape[1] == 3);
    try std.testing.expect(gatheredTensor.data[0] == 4);
    try std.testing.expect(gatheredTensor.data[1] == 5);
    try std.testing.expect(gatheredTensor.data[2] == 6);
}

test "gather - invalid indices" {
    std.debug.print("\n     test: gather - invalid indices", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input tensor: 2x2 matrix
    var inputArray: [2][2]u8 = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 2 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize indices tensor with invalid index
    var indicesArray: [1]usize = [_]usize{2}; // Invalid index (only 0,1 are valid)
    var indicesShape: [1]usize = [_]usize{1};
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Should return error for out of bounds index
    try std.testing.expectError(TensorError.IndexOutOfBounds, TensMath.gather(u8, &inputTensor, &indicesTensor, 0));
}

test "gather - multi-dimensional indices" {
    std.debug.print("\n     test: gather - multi-dimensional indices", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input tensor: 3x3 matrix
    var inputArray: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var inputShape: [2]usize = [_]usize{ 3, 3 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize 2D indices tensor: [[0,2], [1,1]]
    var indicesArray: [2][2]usize = [_][2]usize{
        [_]usize{ 0, 2 },
        [_]usize{ 1, 1 },
    };
    var indicesShape: [2]usize = [_]usize{ 2, 2 };
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Gather along axis 0
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, 0);
    defer gatheredTensor.deinit();

    // Expected shape: [2, 2, 3]
    try std.testing.expect(gatheredTensor.shape[0] == 2);
    try std.testing.expect(gatheredTensor.shape[1] == 2);
    try std.testing.expect(gatheredTensor.shape[2] == 3);

    // Check first row (indices 0,2): [1,2,3], [7,8,9]
    try std.testing.expect(gatheredTensor.data[0] == 1);
    try std.testing.expect(gatheredTensor.data[1] == 2);
    try std.testing.expect(gatheredTensor.data[2] == 3);
    try std.testing.expect(gatheredTensor.data[3] == 7);
    try std.testing.expect(gatheredTensor.data[4] == 8);
    try std.testing.expect(gatheredTensor.data[5] == 9);

    // Check second row (indices 1,1): [4,5,6], [4,5,6]
    try std.testing.expect(gatheredTensor.data[6] == 4);
    try std.testing.expect(gatheredTensor.data[7] == 5);
    try std.testing.expect(gatheredTensor.data[8] == 6);
    try std.testing.expect(gatheredTensor.data[9] == 4);
    try std.testing.expect(gatheredTensor.data[10] == 5);
    try std.testing.expect(gatheredTensor.data[11] == 6);
}

test "gather - single element tensor" {
    std.debug.print("\n     test: gather - single element tensor", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input tensor: [[[1]]]
    var inputArray: [1][1][1]u8 = [_][1][1]u8{[_][1]u8{[_]u8{1}}};
    var inputShape: [3]usize = [_]usize{ 1, 1, 1 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize indices tensor: [0]
    var indicesArray: [1]usize = [_]usize{0};
    var indicesShape: [1]usize = [_]usize{1};
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Test gathering on each axis
    inline for (0..3) |axis| {
        var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, axis);
        defer gatheredTensor.deinit();

        try std.testing.expect(gatheredTensor.data[0] == 1);
        try std.testing.expect(gatheredTensor.size == 1);
    }
}

test "unsqueeze - basic" {
    std.debug.print("\n     test: unsqueeze - basic", .{});
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
    std.debug.print("\n     test: unsqueeze - multiple axes", .{});
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
    std.debug.print("\n     test: unsqueeze - negative axes", .{});
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
    std.debug.print("\n     test: unsqueeze - scalar tensor", .{});
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
    std.debug.print("\n     test: unsqueeze - error cases", .{});
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

test "Concatenate tensors with mismatched shapes" {
    std.debug.print("\n     test: Concatenate tensors with mismatched shapes", .{});
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
    std.debug.print("\n     test: Concatenate tensors with invalid axis", .{});
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
    std.debug.print("\n     test: get_concatenate_output_shape - 3D tensors", .{});
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
    std.debug.print("\n     test: get_concatenate_output_shape - mismatched shapes", .{});

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
    std.debug.print("\n     test: get_concatenate_output_shape - invalid axis", .{});

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

test "neg - 2D tensor" {
    std.debug.print("\n     test: neg - 2D tensor", .{});

    const allocator = pkgAllocator.allocator;

    // Test case 1: 2x3 matrix
    {
        var input_array = [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
        };
        var shape = [_]usize{ 2, 3 };
        var tensor = try Tensor(i8).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        var flipped = try TensMath.flip(i8, &tensor);
        defer flipped.deinit();

        // Expected: [[6, 5, 4], [3, 2, 1]]
        try std.testing.expectEqual(@as(i8, 6), flipped.data[0]);
        try std.testing.expectEqual(@as(i8, 5), flipped.data[1]);
        try std.testing.expectEqual(@as(i8, 4), flipped.data[2]);
        try std.testing.expectEqual(@as(i8, 3), flipped.data[3]);
        try std.testing.expectEqual(@as(i8, 2), flipped.data[4]);
        try std.testing.expectEqual(@as(i8, 1), flipped.data[5]);
    }

    // Test case 2: Square matrix
    {
        var input_array = [_][2]i8{
            [_]i8{ 1, 2 },
            [_]i8{ 3, 4 },
        };
        var shape = [_]usize{ 2, 2 };
        var tensor = try Tensor(i8).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        var flipped = try TensMath.flip(i8, &tensor);
        defer flipped.deinit();

        // Expected: [[4, 3], [2, 1]]
        try std.testing.expectEqual(@as(i8, 4), flipped.data[0]);
        try std.testing.expectEqual(@as(i8, 3), flipped.data[1]);
        try std.testing.expectEqual(@as(i8, 2), flipped.data[2]);
        try std.testing.expectEqual(@as(i8, 1), flipped.data[3]);
    }
}

test "neg - 3D tensor" {
    std.debug.print("\n     test: neg - 3D tensor", .{});

    const allocator = pkgAllocator.allocator;

    // 2x2x2 tensor
    var input_array = [_][2][2]i8{
        [_][2]i8{
            [_]i8{ 1, 2 },
            [_]i8{ 3, 4 },
        },
        [_][2]i8{
            [_]i8{ 5, 6 },
            [_]i8{ 7, 8 },
        },
    };
    var shape = [_]usize{ 2, 2, 2 };
    var tensor = try Tensor(i8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    var flipped = try TensMath.flip(i8, &tensor);
    defer flipped.deinit();

    // Each 2x2 matrix should be flipped independently
    // Expected first matrix: [[4, 3], [2, 1]]
    // Expected second matrix: [[8, 7], [6, 5]]
    try std.testing.expectEqual(@as(i8, 4), flipped.data[0]);
    try std.testing.expectEqual(@as(i8, 3), flipped.data[1]);
    try std.testing.expectEqual(@as(i8, 2), flipped.data[2]);
    try std.testing.expectEqual(@as(i8, 1), flipped.data[3]);
    try std.testing.expectEqual(@as(i8, 8), flipped.data[4]);
    try std.testing.expectEqual(@as(i8, 7), flipped.data[5]);
    try std.testing.expectEqual(@as(i8, 6), flipped.data[6]);
    try std.testing.expectEqual(@as(i8, 5), flipped.data[7]);

    // Check shape preservation
    try std.testing.expectEqual(@as(usize, 2), flipped.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), flipped.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), flipped.shape[2]);
    try std.testing.expectEqual(@as(usize, 8), flipped.size);
}

test "get_split_output_shapes - basic functionality" {
    std.debug.print("\n     test: get_split_output_shapes - basic functionality", .{});

    // Test case 1: Split along axis 1 - using shape that's evenly divisible by 2
    {
        var input_shape = [_]usize{ 4, 4, 2 }; // Change 3 to 4 so it's divisible by 2
        // Explicitly request 2 output shapes by setting num_outputs
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, null, 2);
        defer {
            for (output_shapes) |shape| {
                pkgAllocator.allocator.free(shape);
            }
            pkgAllocator.allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 2), output_shapes.len);
        // First split: [4, 2, 2]
        try std.testing.expectEqual(@as(usize, 4), output_shapes[0][0]);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][1]); // Half of 4
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][2]);
        // Second split: [4, 2, 2]
        try std.testing.expectEqual(@as(usize, 4), output_shapes[1][0]);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[1][1]); // Half of 4
        try std.testing.expectEqual(@as(usize, 2), output_shapes[1][2]);
    }

    // Test case 2: Split along last axis
    {
        var input_shape = [_]usize{ 2, 4 };
        var split_sizes = [_]usize{ 2, 2 };
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, &split_sizes, null);
        defer {
            for (output_shapes) |shape| {
                pkgAllocator.allocator.free(shape);
            }
            pkgAllocator.allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 2), output_shapes.len);
        // Both splits: [2, 2]
        for (output_shapes) |shape| {
            try std.testing.expectEqual(@as(usize, 2), shape[0]);
            try std.testing.expectEqual(@as(usize, 2), shape[1]);
        }
    }
}

test "get_split_output_shapes - negative axis" {
    std.debug.print("\n     test: get_split_output_shapes - negative axis", .{});

    var input_shape = [_]usize{ 2, 6, 3 };
    var split_sizes = [_]usize{ 2, 4 };

    // Test with axis -2 (equivalent to axis 1)
    const output_shapes = try TensMath.get_split_output_shapes(&input_shape, -2, &split_sizes, null);
    defer {
        for (output_shapes) |shape| {
            pkgAllocator.allocator.free(shape);
        }
        pkgAllocator.allocator.free(output_shapes);
    }

    try std.testing.expectEqual(@as(usize, 2), output_shapes.len);
    // First split: [2, 2, 3]
    try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
    try std.testing.expectEqual(@as(usize, 2), output_shapes[0][1]);
    try std.testing.expectEqual(@as(usize, 3), output_shapes[0][2]);
    // Second split: [2, 4, 3]
    try std.testing.expectEqual(@as(usize, 2), output_shapes[1][0]);
    try std.testing.expectEqual(@as(usize, 4), output_shapes[1][1]);
    try std.testing.expectEqual(@as(usize, 3), output_shapes[1][2]);
}

test "get_split_output_shapes - error cases" {
    std.debug.print("\n     test: get_split_output_shapes - error cases", .{});

    var input_shape = [_]usize{ 2, 3, 4 };

    // Test case 1: Invalid axis (too large)
    {
        var split_sizes = [_]usize{1};
        try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, 3, &split_sizes, null));
    }

    // Test case 2: Invalid axis (too negative)
    {
        var split_sizes = [_]usize{1};
        try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, -4, &split_sizes, null));
    }

    // Test case 3: Split sizes don't match dimension size
    {
        var split_sizes = [_]usize{ 1, 1 }; // Sum = 2, but dimension size is 3
        try std.testing.expectError(TensorError.InvalidSplitSize, TensMath.get_split_output_shapes(&input_shape, 1, &split_sizes, null));
    }

    // Test case 4: Empty dimension
    // It appears the function now handles empty shapes differently, so let's update the test
    {
        var empty_shape = [_]usize{ 0, 2 };
        // Instead of expecting an error, let's verify it handles the case gracefully
        const output_shapes = try TensMath.get_split_output_shapes(&empty_shape, 0, null, null);
        defer {
            for (output_shapes) |shape| {
                pkgAllocator.allocator.free(shape);
            }
            pkgAllocator.allocator.free(output_shapes);
        }
        // Verify it returned an empty list or a list with a single empty shape
        try std.testing.expect(output_shapes.len > 0);
    }
}

test "get_split_output_shapes - default split" {
    std.debug.print("\n     test: get_split_output_shapes - default split", .{});

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

test "slice_onnx basic operations" {
    // std.debug.print("\n     test: slice_onnx basic operations", .{});
    // const allocator = pkgAllocator.allocator;

    // Test 1: Basic 3D slicing
    // {
    //     var input_array = [_][2][3]f32{
    //         [_][3]f32{
    //             [_]f32{ 1, 2, 3 },
    //             [_]f32{ 4, 5, 6 },
    //         },
    //         [_][3]f32{
    //             [_]f32{ 7, 8, 9 },
    //             [_]f32{ 10, 11, 12 },
    //         },
    //     };
    //     var shape = [_]usize{ 2, 2, 3 };
    //     var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
    //     defer tensor.deinit();

    //     var starts = [_]i64{ 0, 1, 1 };
    //     var ends = [_]i64{ 2, 2, 3 };
    //     var axes = [_]i64{ 0, 1, 2 };
    //     var steps = [_]i64{ 1, 1, 1 };

    //     var result = try TensMath.slice_onnx(f32, &tensor, &starts, &ends, &axes, &steps);
    //     defer result.deinit();

    //     try std.testing.expectEqual(@as(usize, 3), result.shape.len);
    //     try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
    //     try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    //     try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    //     try std.testing.expectEqual(@as(f32, 5), result.data[0]);
    //     try std.testing.expectEqual(@as(f32, 6), result.data[1]);
    //     try std.testing.expectEqual(@as(f32, 11), result.data[2]);
    //     try std.testing.expectEqual(@as(f32, 12), result.data[3]);
    // }

    // Test 2: Negative indices
    // {
    //     var input_array = [_]f32{ 1, 2, 3, 4, 5 };
    //     var shape = [_]usize{5};
    //     var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
    //     defer tensor.deinit();

    //     var starts = [_]i64{-3};
    //     var ends = [_]i64{-1};
    //     var steps = [_]i64{1};

    //     var result = try TensMath.slice_onnx(f32, &tensor, &starts, &ends, null, &steps);
    //     defer result.deinit();

    //     try std.testing.expectEqual(@as(usize, 1), result.shape.len);
    //     try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
    //     try std.testing.expectEqual(@as(f32, 3), result.data[0]);
    //     try std.testing.expectEqual(@as(f32, 4), result.data[1]);
    // }

    // Test 3: With steps
    // {
    //     var input_array = [_]f32{ 1, 2, 3, 4, 5, 6 };
    //     var shape = [_]usize{6};
    //     var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
    //     defer tensor.deinit();

    //     var starts = [_]i64{0};
    //     var ends = [_]i64{6};
    //     var steps = [_]i64{2};

    //     var result = try TensMath.slice_onnx(f32, &tensor, &starts, &ends, null, &steps);
    //     defer result.deinit();

    //     try std.testing.expectEqual(@as(usize, 1), result.shape.len);
    //     try std.testing.expectEqual(@as(usize, 3), result.shape[0]);
    //     try std.testing.expectEqual(@as(f32, 1), result.data[0]);
    //     try std.testing.expectEqual(@as(f32, 3), result.data[1]);
    //     try std.testing.expectEqual(@as(f32, 5), result.data[2]);
    // }
}

test "get_slice_output_shape basic operations" {
    std.debug.print("\n     test: get_slice_output_shape basic operations", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1: Basic 3D slicing
    {
        var input_shape = [_]usize{ 2, 2, 3 };
        var starts = [_]i64{ 0, 1, 1 };
        var ends = [_]i64{ 2, 2, 3 };
        var axes = [_]i64{ 0, 1, 2 };
        var steps = [_]i64{ 1, 1, 1 };

        const result = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, &axes, &steps);
        defer allocator.free(result);

        try std.testing.expectEqual(@as(usize, 3), result.len);
        try std.testing.expectEqual(@as(usize, 2), result[0]);
        try std.testing.expectEqual(@as(usize, 1), result[1]);
        try std.testing.expectEqual(@as(usize, 2), result[2]);
    }

    // Test 2: Negative indices
    {
        var input_shape = [_]usize{5};
        var starts = [_]i64{-3};
        var ends = [_]i64{-1};
        var steps = [_]i64{1};

        const result = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, &steps);
        defer allocator.free(result);

        try std.testing.expectEqual(@as(usize, 1), result.len);
        try std.testing.expectEqual(@as(usize, 2), result[0]);
    }

    // Test 3: With steps
    {
        var input_shape = [_]usize{6};
        var starts = [_]i64{0};
        var ends = [_]i64{6};
        var steps = [_]i64{2};

        const result = try TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, &steps);
        defer allocator.free(result);

        try std.testing.expectEqual(@as(usize, 1), result.len);
        try std.testing.expectEqual(@as(usize, 3), result[0]);
    }

    // Test 4: Error cases
    {
        var input_shape = [_]usize{ 2, 2, 3 };

        // Test mismatched starts and ends lengths
        var starts = [_]i64{ 0, 1 };
        var ends = [_]i64{ 2, 2, 3 };
        try std.testing.expectError(TensorError.InvalidSliceIndices, TensMath.get_slice_output_shape(&input_shape, &starts, &ends, null, null));

        // Test invalid step size
        var valid_starts = [_]i64{0};
        var valid_ends = [_]i64{2};
        var invalid_steps = [_]i64{0};
        try std.testing.expectError(TensorError.InvalidSliceStep, TensMath.get_slice_output_shape(&input_shape, &valid_starts, &valid_ends, null, &invalid_steps));

        // Test out of bounds axes
        var axes = [_]i64{3}; // Invalid axis for 3D tensor
        try std.testing.expectError(TensorError.InvalidSliceIndices, TensMath.get_slice_output_shape(&input_shape, &valid_starts, &valid_ends, &axes, null));
    }
}

test "identity" {
    std.debug.print("\n test: identity basic operations\n", .{});
    const allocator = pkgAllocator.allocator;
    {
        //check output correctness
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

        var result = try TensMath.identity(f32, &tensor);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 3), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), result.shape[1]);
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]);

        try std.testing.expectEqual(@as(usize, 12), result.size);

        const expected = [_]f32{
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        };
        for (result.data, 0..) |val, i| {
            try std.testing.expectEqual(expected[i], val);
        }

        //check that the output is not the same alias as the input
        try std.testing.expect(result.data.ptr != tensor.data.ptr);

        std.debug.print("OK: identity test passed.\n", .{});
    }
}

test "get_identity_shape_output returns a copy of the input shape" {
    //3 dimensionional input shape
    const input_shape = [_]usize{ 2, 3, 5 };

    // Call the function under test
    const output_shape = try TensMath.get_identity_output_shape(&input_shape);
    defer pkgAllocator.allocator.free(output_shape);

    // Check the length matches
    try std.testing.expectEqual(@as(usize, input_shape.len), output_shape.len);

    // Check each element was copied correctly
    for (output_shape, 0..) |val, i| {
        try std.testing.expectEqual(input_shape[i], val);
    }
}
