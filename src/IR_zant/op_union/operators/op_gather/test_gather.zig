const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "gather along axis 0" {
    const allocator = pkgAllocator.allocator;

    tests_log.info("\n     test: gather along axis 0", .{});

    // Initialize input tensor: 3x3 matrix
    var inputArray: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var inputShape: [2]usize = [_]usize{ 3, 3 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize indices tensor: [0, 2]
    var indicesArray: [2]usize = [_]usize{ 0, 2 };
    var indicesShape: [1]usize = [_]usize{2};
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Perform gather along axis 0
    const validAxis: isize = 0;
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, validAxis);
    defer gatheredTensor.deinit();

    // Expected output tensor: [1,2,3,7,8,9], shape [2,3]
    const expectedData: [6]u8 = [_]u8{ 1, 2, 3, 7, 8, 9 };
    const expectedShape: [2]usize = [_]usize{ 2, 3 };

    // Check shape
    try std.testing.expect(gatheredTensor.shape.len == expectedShape.len);
    for (0..expectedShape.len) |i| {
        try std.testing.expect(gatheredTensor.shape[i] == expectedShape[i]);
    }

    // Check data
    try std.testing.expect(gatheredTensor.size == 6);
    for (0..gatheredTensor.size) |i| {
        try std.testing.expect(gatheredTensor.data[i] == expectedData[i]);
    }
}

test "gather along axis 1" {
    const allocator = pkgAllocator.allocator;

    tests_log.info("\n     test: gather along axis 1", .{});

    var inputArray: [2][4]u8 = [_][4]u8{
        [_]u8{ 10, 20, 30, 40 },
        [_]u8{ 50, 60, 70, 80 },
    };
    var inputShape: [2]usize = [_]usize{ 2, 4 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    var indicesArray: [2][2]usize = [_][2]usize{
        [_]usize{ 1, 3 },
        [_]usize{ 0, 2 },
    };
    var indicesShape: [2]usize = [_]usize{ 2, 2 };
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    // Perform gather along axis 1
    const validAxis: isize = 1;
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, validAxis);
    defer gatheredTensor.deinit();

    // Expected output tensor: [
    //   [20, 40],
    //   [10, 30],
    //   [60, 80],
    //   [50, 70]
    // ], shape [2, 2, 2]
    const expectedData: [8]u8 = [_]u8{ 20, 40, 10, 30, 60, 80, 50, 70 };
    const expectedShape: [3]usize = [_]usize{ 2, 2, 2 };

    // Check shape
    try std.testing.expect(gatheredTensor.shape.len == expectedShape.len);
    for (0..expectedShape.len) |i| {
        try std.testing.expect(gatheredTensor.shape[i] == expectedShape[i]);
    }

    // Check data
    tests_log.debug("\n     gatheredTensor.size: {}\n", .{gatheredTensor.size});
    gatheredTensor.print();

    try std.testing.expect(gatheredTensor.size == 8);
    for (0..gatheredTensor.size) |i| {
        tests_log.debug("\n     gatheredTensor.data[i]: {}\n", .{expectedData[i]});
        tests_log.debug("\n     expectedData[i]: {}\n", .{gatheredTensor.data[i]});
        try std.testing.expect(gatheredTensor.data[i] == expectedData[i]);
    }
}

test "gather with invalid axis" {
    const allocator = pkgAllocator.allocator;

    tests_log.info("\n     test: gather with invalid axis", .{});

    // Initialize input tensor: 3x3 matrix
    var inputArray: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var inputShape: [2]usize = [_]usize{ 3, 3 };
    var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
    defer inputTensor.deinit();

    // Initialize indices tensor: [0, 2]
    var indicesArray: [2]usize = [_]usize{ 0, 2 };
    var indicesShape: [1]usize = [_]usize{2};
    var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
    defer indicesTensor.deinit();

    const invalidAxis: isize = 3; // Input tensor has 2 dimensions
    const result = TensMath.gather(u8, &inputTensor, &indicesTensor, invalidAxis);
    try std.testing.expect(result == TensorError.InvalidAxis);
}

test "gather - negative axis" {
    tests_log.info("\n     test: gather - negative axis", .{});
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
    const validAxis: isize = -2;
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, validAxis);
    defer gatheredTensor.deinit();

    // Expected: [4, 5, 6]
    try std.testing.expect(gatheredTensor.shape[0] == 1);
    try std.testing.expect(gatheredTensor.shape[1] == 3);
    try std.testing.expect(gatheredTensor.data[0] == 4);
    try std.testing.expect(gatheredTensor.data[1] == 5);
    try std.testing.expect(gatheredTensor.data[2] == 6);
}

test "gather - invalid indices" {
    tests_log.info("\n     test: gather - invalid indices", .{});
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
    const validAxis: isize = 0;
    try std.testing.expectError(TensorError.IndexOutOfBounds, TensMath.gather(u8, &inputTensor, &indicesTensor, validAxis));
}

test "gather - multi-dimensional indices" {
    tests_log.info("\n     test: gather - multi-dimensional indices", .{});
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
    const validAxis: isize = 0;
    var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, validAxis);
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
    tests_log.info("\n     test: gather - single element tensor", .{});
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
