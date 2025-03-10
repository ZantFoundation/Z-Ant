const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const PoolingType = zant.model.layer.poolingLayer.PoolingType;
const AutoPadType = zant.core.tensor.math_standard.AutoPadType;

test "Pooling 2D" {
    std.debug.print("\n     test: Pooling 2D\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3 input, same as original
    var shape_tensor: [2]usize = [_]usize{ 3, 3 };
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    // Calculate W = number of windows
    const input_rows = input1.shape[0];
    const input_cols = input1.shape[1];
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; // 2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; // 2
    const W = out_rows * out_cols; // 4

    // Instead of used_input being shape of input, now [W,3,3]
    var used_input1_shape = [_]usize{ W, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // Same checks for output as original
    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    // Now we have 4 windows (W=4), each 3x3
    // The original code expected certain pattern in used_input:
    // (1,1)=5 for first window
    // (1,2)=6 for second window
    // (2,1)=50 for third window
    // (2,2)=60 for fourth window
    //
    // Windows:
    // w=0: top-left max at (1,1)
    // w=1: top-right max at (1,2)
    // w=2: bottom-left max at (2,1)
    // w=3: bottom-right max at (2,2)

    // Check w=0 window:
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 1 * 3 + 1]);

    // w=1: top-right (1,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[1 * 9 + 1 * 3 + 2]);

    // w=2: bottom-left (2,1)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[2 * 9 + 2 * 3 + 1]);

    // w=3: bottom-right (2,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[3 * 9 + 2 * 3 + 2]);

    // Another test with kernel2 and stride2
    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    const out_rows_2 = (3 - 2 + 1) / 2;
    const out_cols_2 = (3 - 2 + 1) / 2;
    const W2 = out_rows_2 * out_cols_2;

    var used_input2_shape = [_]usize{ W2, 3, 3 };
    var used_input2 = try Tensor(u8).fromShape(&allocator, used_input2_shape[0..]);
    defer used_input2.deinit();
    for (used_input2.data) |*val| val.* = 0;

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);

    // single window w=0 max at (1,1)
    try std.testing.expectEqual(@as(u1, 1), used_input2.data[0 * (3 * 3) + 1 * 3 + 1]);
}

test "Pooling multidim" {
    std.debug.print("\n     test: Pooling multidim\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3x3 input as original
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    const d = input1.shape[0]; // 3
    const input_rows = input1.shape[1]; //3
    const input_cols = input1.shape[2]; //3
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; //2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; //2
    const W_tot = d * out_rows * out_cols; // 3*2*2=12

    var used_input1_shape = [_]usize{ W_tot, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);

    // We do not check used_input details here, just shapes as in original.
}

test "ONNX MaxPool - NOTSET padding" {
    std.debug.print("\n     test: ONNX MaxPool - NOTSET padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // top, left, bottom, right

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 6), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 14), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 16), result.output.data[3]);
}

test "ONNX MaxPool - SAME_UPPER padding" {
    std.debug.print("\n     test: ONNX MaxPool - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 }; // not used in SAME_UPPER

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .SAME_UPPER,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 5), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[3]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[4]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[5]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[6]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[7]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[8]);
}

test "ONNX MaxPool - with dilation" {
    std.debug.print("\n     test: ONNX MaxPool - with dilation\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x4x4
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 2, 2 }; // Dilated kernel
    var pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    // With dilation=2, each kernel element skips one position
    // So kernel covers positions: [[1,3],[9,11]] for first window
    try std.testing.expectEqual(@as(f32, 11), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 12), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 15), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 16), result.output.data[3]);
}

test "ONNX MaxPool - ceil mode" {
    std.debug.print("\n     test: ONNX MaxPool - ceil mode\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 2, 2 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        true, // ceil_mode = true
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[2]); // Ceil mode makes this 2 instead of 1
    try std.testing.expectEqual(@as(usize, 2), result.output.shape[3]);

    try std.testing.expectEqual(@as(f32, 5), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[3]);
}

test "ONNX MaxPool - explicit padding" {
    std.debug.print("\n     test: ONNX MaxPool - explicit padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: 1x1x3x3
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var kernel_shape = [_]usize{ 2, 2 };
    var strides = [_]usize{ 1, 1 };
    var dilations = [_]usize{ 1, 1 };
    var pads = [_]usize{ 1, 1, 1, 1 }; // pad 1 on all sides

    var result = try TensMath.onnx_maxpool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
    );
    defer {
        result.output.deinit();
        result.used_input.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.output.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), result.output.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), result.output.shape[3]);

    // First row includes padding (0s)
    try std.testing.expectEqual(@as(f32, 1), result.output.data[0]);
    try std.testing.expectEqual(@as(f32, 2), result.output.data[1]);
    try std.testing.expectEqual(@as(f32, 3), result.output.data[2]);
    try std.testing.expectEqual(@as(f32, 3), result.output.data[3]);

    // Middle rows
    try std.testing.expectEqual(@as(f32, 4), result.output.data[4]);
    try std.testing.expectEqual(@as(f32, 5), result.output.data[5]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[6]);
    try std.testing.expectEqual(@as(f32, 6), result.output.data[7]);

    try std.testing.expectEqual(@as(f32, 7), result.output.data[8]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[9]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[10]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[11]);

    // Last row includes padding (0s)
    try std.testing.expectEqual(@as(f32, 7), result.output.data[12]);
    try std.testing.expectEqual(@as(f32, 8), result.output.data[13]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[14]);
    try std.testing.expectEqual(@as(f32, 9), result.output.data[15]);
}
