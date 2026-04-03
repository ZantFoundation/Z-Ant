const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ErrorHandler = error_handler;

const tests_log = std.log.scoped(.test_gemm);

// TODO: add test for multiple batch/channel

test "Gemm Y = a A*B" {
    tests_log.info("\n     test: Gemm Y = a A*B", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [4]usize = [_]usize{ 1, 1, 2, 2 }; // 1 batch, 1 channel, 2x2 matrix

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

    var result_tensor = try TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false);

    try std.testing.expect(9.0 == result_tensor.data[0]);
    try std.testing.expect(12.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
}

test "Gemm Y = a A*B + bC without broadcasting" {
    tests_log.info("\n     test: Gemm Y = a A*B + bC without broadcasting", .{});

    const allocator = pkgAllocator.allocator;

    var a_shape: [4]usize = [_]usize{ 1, 1, 2, 2 }; // 1 batch, 1 channel, 2x2 matrix
    var b_shape: [4]usize = [_]usize{ 1, 1, 2, 1 }; // 1 batch, 1 channel, 2x1 matrix

    var inputArrayA: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArrayB: [2][1]f32 = [_][1]f32{
        [_]f32{1.0},
        [_]f32{4.0},
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArrayA, &a_shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);
    var t3 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);

    var result_tensor = try TensMath.gemm(f32, &t1, &t2, &t3, 1, 1, false, false);

    try std.testing.expect(10.0 == result_tensor.data[0]);
    try std.testing.expect(28.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
    t3.deinit();
}

test "Gemm Y = a A*B + bC with broadcasting" {
    tests_log.info("\n     test: Gemm Y = a A*B + bC with broadcasting", .{});

    const allocator = pkgAllocator.allocator;

    var a_shape: [4]usize = [_]usize{ 1, 1, 3, 3 }; // 1 batch, 1 channel, 3x3 matrix
    var b_shape: [4]usize = [_]usize{ 1, 1, 3, 2 }; // 1 batch, 1 channel, 3x2 matrix
    var c_shape: [4]usize = [_]usize{ 1, 1, 1, 2 }; // 1 batch, 1 channel, 1x2 matrix

    var inputArrayA: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 7.0, 8.0, 9.0 },
    };
    var inputArrayB: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 7.0, 8.0 },
    };
    var inputArrayC: [1][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArrayA, &a_shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);
    var t3 = try Tensor(f32).fromArray(&allocator, &inputArrayC, &c_shape);

    var result_tensor = try TensMath.gemm(f32, &t1, &t2, &t3, 1, 1, false, false);

    try std.testing.expect(31.0 == result_tensor.data[0]);
    try std.testing.expect(128.0 == result_tensor.data[5]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
    t3.deinit();
}

test "Gemm Y = a A*B + bC with broadcasting, custom parameters v1" {
    tests_log.info("\n     test: Gemm Y = a A*B + bC with broadcasting, custom parameters v1", .{});

    const allocator = pkgAllocator.allocator;

    var a_shape: [4]usize = [_]usize{ 1, 1, 3, 3 }; // 1 batch, 1 channel, 3x3 matrix
    var b_shape: [4]usize = [_]usize{ 1, 1, 2, 3 }; // 1 batch, 1 channel, 2x3 matrix
    var c_shape: [4]usize = [_]usize{ 1, 1, 1, 2 }; // 1 batch, 1 channel, 1x2 matrix

    var inputArrayA: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 7.0, 8.0, 9.0 },
    };
    var inputArrayB: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 4.0, 7.0 },
        [_]f32{ 2.0, 5.0, 8.0 },
    };
    var inputArrayC: [1][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArrayA, &a_shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);
    var t3 = try Tensor(f32).fromArray(&allocator, &inputArrayC, &c_shape);

    var result_tensor = try TensMath.gemm(f32, &t1, &t2, &t3, 2, 3, false, true);

    try std.testing.expect(63.0 == result_tensor.data[0]);
    try std.testing.expect(258.0 == result_tensor.data[5]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
    t3.deinit();
}

test "Gemm Y = a A*B + bC with broadcasting, custom parameters v2" {
    tests_log.info("\n     test: Gemm Y = a A*B + bC with broadcasting, custom parameters v2", .{});

    const allocator = pkgAllocator.allocator;

    var a_shape: [4]usize = [_]usize{ 1, 1, 3, 2 }; // 1 batch, 1 channel, 3x3 matrix
    var b_shape: [4]usize = [_]usize{ 1, 1, 2, 3 }; // 1 batch, 1 channel, 2x3 matrix
    var c_shape: [4]usize = [_]usize{ 1, 1, 2, 1 }; // 1 batch, 1 channel, 1x2 matrix

    var inputArrayA: [3][2]f32 = [_][2]f32{
        [_]f32{
            1.0,
            2.0,
        },
        [_]f32{
            4.0,
            5.0,
        },
        [_]f32{
            7.0,
            8.0,
        },
    };
    var inputArrayB: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 4.0, 7.0 },
        [_]f32{ 2.0, 5.0, 8.0 },
    };
    var inputArrayC: [2][1]f32 = [_][1]f32{
        [_]f32{1.0},
        [_]f32{2.0},
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArrayA, &a_shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);
    var t3 = try Tensor(f32).fromArray(&allocator, &inputArrayC, &c_shape);

    var result_tensor = try TensMath.gemm(f32, &t1, &t2, &t3, 2, 3, true, true);

    try std.testing.expect(135.0 == result_tensor.data[0]);
    try std.testing.expect(159.0 == result_tensor.data[1]);
    try std.testing.expect(162.0 == result_tensor.data[2]);
    try std.testing.expect(192.0 == result_tensor.data[3]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
    t3.deinit();
}

// NOTE: as 22/02 this test is not passed as mat_mul, used by gemm, doesn't support multiplication of matrix distribuited in multiple batches/channels but only tensor with a shape like {1, 1, N, M}, once mat_mul is updated this test should pass
// test "Gemm Y = a A*B + bC with broadcasting, custom parameters, multiple batch/channels" {
//     tests_log.info("\n     test: Gemm Y = a A*B + bC with broadcasting, custom parameters, multiple batch/channels", .{});

//     const allocator = pkgAllocator.allocator;

//     var a_shape: [4]usize = [_]usize{ 2, 2, 3, 2 }; // 2 batch, 2 channel, 3x2 matrix
//     var b_shape: [4]usize = [_]usize{ 2, 2, 2, 3 }; // 2 batch, 2 channel, 2x3 matrix
//     var c_shape: [4]usize = [_]usize{ 2, 2, 2, 1 }; // 2 batch, 2 channel, 2x1 matrix

//     // 2 batch, 2 channel, 3x2 matrix
//     var inputArrayA: [2][2][3][2]f32 = [_][2][3][2]f32{
//         // Batch 0
//         [_][3][2]f32{
//             // Channel 0
//             [_][2]f32{ [_]f32{ 1.0, 2.0 }, [_]f32{ 3.0, 4.0 }, [_]f32{ 5.0, 6.0 } },
//             // Channel 1
//             [_][2]f32{ [_]f32{ 7.0, 8.0 }, [_]f32{ 9.0, 10.0 }, [_]f32{ 11.0, 12.0 } },
//         },
//         // Batch 1
//         [_][3][2]f32{
//             // Channel 0
//             [_][2]f32{ [_]f32{ 13.0, 14.0 }, [_]f32{ 15.0, 16.0 }, [_]f32{ 17.0, 18.0 } },
//             // Channel 1
//             [_][2]f32{ [_]f32{ 19.0, 20.0 }, [_]f32{ 21.0, 22.0 }, [_]f32{ 23.0, 24.0 } },
//         },
//     };

//     // 2 batch, 2 channel, 2x3 matrix
//     var inputArrayB: [2][2][2][3]f32 = [_][2][2][3]f32{
//         // Batch 0
//         [_][2][3]f32{
//             // Channel 0
//             [_][3]f32{ [_]f32{ 1.0, 2.0, 3.0 }, [_]f32{ 4.0, 5.0, 6.0 } },
//             // Channel 1
//             [_][3]f32{ [_]f32{ 7.0, 8.0, 9.0 }, [_]f32{ 10.0, 11.0, 12.0 } },
//         },
//         // Batch 1
//         [_][2][3]f32{
//             // Channel 0
//             [_][3]f32{ [_]f32{ 13.0, 14.0, 15.0 }, [_]f32{ 16.0, 17.0, 18.0 } },
//             // Channel 1
//             [_][3]f32{ [_]f32{ 19.0, 20.0, 21.0 }, [_]f32{ 22.0, 23.0, 24.0 } },
//         },
//     };

//     // 2 batch, 2 channel, 2x1 matrix
//     var inputArrayC: [2][2][2][1]f32 = [_][2][2][1]f32{
//         // Batch 0
//         [_][2][1]f32{
//             // Channel 0
//             [_][1]f32{ [_]f32{1.0}, [_]f32{2.0} },
//             // Channel 1
//             [_][1]f32{ [_]f32{3.0}, [_]f32{4.0} },
//         },
//         // Batch 1
//         [_][2][1]f32{
//             // Channel 0
//             [_][1]f32{ [_]f32{5.0}, [_]f32{6.0} },
//             // Channel 1
//             [_][1]f32{ [_]f32{7.0}, [_]f32{8.0} },
//         },
//     };

//     var t1 = try Tensor(f32).fromArray(&allocator, &inputArrayA, &a_shape);
//     var t2 = try Tensor(f32).fromArray(&allocator, &inputArrayB, &b_shape);
//     var t3 = try Tensor(f32).fromArray(&allocator, &inputArrayC, &c_shape);

//     var result_tensor = try TensMath.gemm(f32, &t1, &t2, &t3, 2, 3, true, true);

//     debug
//     for (0..result_tensor.data.len) |i|
//         tests_log.info("\nres[{d}] {d}", .{ i, result_tensor.data[i] });

//     try std.testing.expect(2549.0 == result_tensor.data[12]);
//     try std.testing.expect(2927.0 == result_tensor.data[13]);
//     try std.testing.expect(2672.0 == result_tensor.data[14]);
//     try std.testing.expect(3068.0 == result_tensor.data[15]);

//     result_tensor.deinit();
//     t1.deinit();
//     t2.deinit();
//     t3.deinit();
// }

test "Error when input tensors aren't 4D" {
    tests_log.info("\n     test: Error when input tensors aren't 4D", .{});

    const allocator = pkgAllocator.allocator;

    var shape1: [3]usize = [_]usize{ 1.0, 2.0, 2.0 }; // missing batch, 1 batch, 2x2 matrix
    var shape2: [3]usize = [_]usize{ 1.0, 2.0, 2.0 }; // missing batch, 1 batch, 2x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false));

    _ = TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false) catch |err| {
        tests_log.warn("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when there's a mismatch in batch or channel dimension" {
    tests_log.info("\n     test: Error when there's a mismatch in batch or channel dimension", .{});

    const allocator = pkgAllocator.allocator;

    var shape1: [4]usize = [_]usize{ 2.0, 1.0, 2.0, 2.0 }; // batch/channel mismatch, 2x2 matrix
    var shape2: [4]usize = [_]usize{ 1.0, 2.0, 2.0, 2.0 }; // batch/channel mismatch, 2x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentShape, TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false));

    _ = TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false) catch |err| {
        tests_log.warn("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible sizes for gemm" {
    tests_log.info("\n     test: Error when input tensors have incompatible sizes for gemm", .{});

    const allocator = pkgAllocator.allocator;

    var shape1: [4]usize = [_]usize{ 4, 4, 2, 2 }; // 1 batch, 1 channel, 2x2 matrix, not compatible for product
    var shape2: [4]usize = [_]usize{ 4, 4, 3, 2 }; // 1 batch, 1 channel, 3x2 matrix, not compatible for product
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDimensionMismatch, TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false));

    _ = TensMath.gemm(f32, &t1, &t2, null, 1, 1, false, false) catch |err| {
        tests_log.warn("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}
