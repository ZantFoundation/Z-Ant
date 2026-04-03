const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "test neg() " {
    tests_log.info("\n     test: neg()", .{});

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
    tests_log.debug("TRY WITH THISSS: \n", .{});
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

test "neg - 2D tensor" {
    tests_log.info("\n     test: neg - 2D tensor", .{});

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
    tests_log.info("\n     test: neg - 3D tensor", .{});

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

// test "lowerNeg - print Uops sequence" {
//     std.debug.print("\n test: lowerNeg - print Uops sequence\n", .{});
//     const allocator = pkgAllocator.allocator;
//     var b = UOpBuilder.init(allocator);
//     defer b.deinit();

//     var input_array: [1][2][2][3]f32 = [_][2][2][3]f32{
//         [_][2][3]f32{
//             [_][3]f32{
//                 [_]f32{ 1, 2, 3 },
//                 [_]f32{ 4, 5, 6 },
//             },
//             [_][3]f32{
//                 [_]f32{ 7, 8, 9 },
//                 [_]f32{ 10, 11, 12 },
//             },
//         },
//     };
//     var shape = [_]usize{ 1, 2, 2, 3 };
//     var strides = [_]isize{ 12, 6, 3, 1 };
//     var input_tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
//     defer input_tensor.deinit();

//     const X_id = b.push(.DEFINE_GLOBAL, .f32, &.{}, Any{ .shape = &shape });

//     _ = lowerNeg(
//         &b,
//         X_id,
//         &strides,
//         &shape,
//         .f32,
//     );

//     std.debug.print("\nUOps sequence:\n", .{});
//     for (b.list.items, 0..) |op, i| {
//         std.debug.print("{d:3}: {s}\n", .{ i, @tagName(op.op) });
//     }
// }
