const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const PoolingType = zant.core.tensor.math_standard.PoolingType;
const AutoPadType = zant.core.tensor.math_standard.AutoPadType;

const tests_log = std.log.scoped(.test_pooling);

const cgv2 = @import("codegen").codegen_v2;
const Uops = cgv2.uops;
const UOpBuilder = cgv2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

const IR = @import("IR_zant");
const MaxPool = IR.operators.MaxPool;
const lowerMaxPool2d = MaxPool.lowerMaxPool2d;

test "ONNX MaxPool - NOTSET padding" {
    tests_log.info("\n     test: ONNX MaxPool - NOTSET padding\n", .{});

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
    tests_log.info("\n     test: ONNX MaxPool - SAME_UPPER padding\n", .{});

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
    tests_log.info("\n     test: ONNX MaxPool - with dilation\n", .{});

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

// test "ONNX MaxPool - ceil mode" {
//     tests_log.info("\n     test: ONNX MaxPool - ceil mode\n", .{});

//     const allocator = pkgAllocator.allocator;

//     // Input: 1x1x3x3
//     var input_shape = [_]usize{ 1, 1, 3, 3 };
//     var input_data = [_]f32{
//         1, 2, 3,
//         4, 5, 6,
//         7, 8, 9,
//     };

//     var input = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
//     defer input.deinit();

//     var kernel_shape = [_]usize{ 2, 2 };
//     var strides = [_]usize{ 2, 2 };
//     var dilations = [_]usize{ 1, 1 };
//     var pads = [_]usize{ 0, 0, 0, 0 };

//     var result = try TensMath.onnx_maxpool(
//         f32,
//         &input,
//         &kernel_shape,
//         &strides,
//         &dilations,
//         &pads,
//         .NOTSET,
//         true, // ceil_mode = true
//     );
//     defer {
//         result.output.deinit();
//         result.used_input.deinit();
//     }

//     try std.testing.expectEqual(@as(f32, 5), result.output.data[0]);
//     try std.testing.expectEqual(@as(f32, 6), result.output.data[1]);
//     try std.testing.expectEqual(@as(f32, 8), result.output.data[2]);
//     try std.testing.expectEqual(@as(f32, 9), result.output.data[3]);
// }

test "ONNX MaxPool - explicit padding" {
    tests_log.info("\n     test: ONNX MaxPool - explicit padding\n", .{});

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

test "ONNX AveragePool - NOTSET padding" {
    tests_log.info("\n     test: ONNX AveragePool - NOTSET padding\n", .{});

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

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        false, // count_include_pad = false
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.5), output.data[0]); // (1+2+5+6)/4
    try std.testing.expectEqual(@as(f32, 5.5), output.data[1]); // (3+4+7+8)/4
    try std.testing.expectEqual(@as(f32, 11.5), output.data[2]); // (9+10+13+14)/4
    try std.testing.expectEqual(@as(f32, 13.5), output.data[3]); // (11+12+15+16)/4
}

test "ONNX AveragePool - SAME_UPPER padding" {
    tests_log.info("\n     test: ONNX AveragePool - SAME_UPPER padding\n", .{});

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
    var pads = [_]usize{ 0, 0, 0, 0 }; // non usati con SAME_UPPER

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .SAME_UPPER,
        false,
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.0), output.data[1]); // (2+3+5+6)/4
    try std.testing.expectEqual(@as(f32, 4.5), output.data[2]); // (3+6+0+0)/2
    try std.testing.expectEqual(@as(f32, 6.0), output.data[3]); // (4+5+7+8)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[4]); // (5+6+8+9)/4
    try std.testing.expectEqual(@as(f32, 7.5), output.data[5]); // (6+9+0+0)/2
    try std.testing.expectEqual(@as(f32, 7.5), output.data[6]); // (7+8+0+0)/2
    try std.testing.expectEqual(@as(f32, 8.5), output.data[7]); // (8+9+0+0)/2
    try std.testing.expectEqual(@as(f32, 9.0), output.data[8]); // (9+0+0+0)/1
}

test "ONNX AveragePool - with dilation" {
    tests_log.info("\n     test: ONNX AveragePool - with dilation\n", .{});

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
    var dilations = [_]usize{ 2, 2 };
    var pads = [_]usize{ 0, 0, 0, 0 };

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 6.0), output.data[0]); // (1+3+9+11)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[1]); // (2+4+10+12)/4
    try std.testing.expectEqual(@as(f32, 10.0), output.data[2]); // (5+7+13+15)/4
    try std.testing.expectEqual(@as(f32, 11.0), output.data[3]); // (6+8+14+16)/4
}

test "ONNX AveragePool - ceil mode" {
    tests_log.info("\n     test: ONNX AveragePool - ceil mode\n", .{});

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

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        true, // ceil_mode = true
        false,
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.5), output.data[1]); // (3+6)/2
    try std.testing.expectEqual(@as(f32, 7.5), output.data[2]); // (7+8)/2
    try std.testing.expectEqual(@as(f32, 9.0), output.data[3]); // (9)/1
}

test "ONNX AveragePool - explicit padding with count_include_pad" {
    tests_log.info("\n     test: ONNX AveragePool - explicit padding with count_include_pad\n", .{});

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
    var pads = [_]usize{ 0, 0, 1, 1 }; // bottom, right padding

    var output = try TensMath.onnx_averagepool(
        f32,
        &input,
        &kernel_shape,
        &strides,
        &dilations,
        &pads,
        .NOTSET,
        false,
        true, // count_include_pad = true
    );
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[3]);

    try std.testing.expectEqual(@as(f32, 3.0), output.data[0]); // (1+2+4+5)/4
    try std.testing.expectEqual(@as(f32, 4.0), output.data[1]); // (2+3+5+6)/4
    try std.testing.expectEqual(@as(f32, 2.25), output.data[2]); // (3+6+0+0)/4
    try std.testing.expectEqual(@as(f32, 6.0), output.data[3]); // (4+5+7+8)/4
    try std.testing.expectEqual(@as(f32, 7.0), output.data[4]); // (5+6+8+9)/4
    try std.testing.expectEqual(@as(f32, 3.75), output.data[5]); // (6+9+0+0)/4
    try std.testing.expectEqual(@as(f32, 3.75), output.data[6]); // (7+8+0+0)/4
    try std.testing.expectEqual(@as(f32, 4.25), output.data[7]); // (8+9+0+0)/4
    try std.testing.expectEqual(@as(f32, 2.25), output.data[8]); // (9+0+0+0)/4
}

test "Manual AveragePool Test" {
    const T = f32;
    var allocator = std.testing.allocator;

    // Input tensor [1, 1, 4, 4]
    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    var input = try Tensor(T).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    // Output tensor [1, 1, 3, 3]
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output = try Tensor(T).fromShape(&allocator, &output_shape);
    defer output.deinit();

    const kernel_shape = [_]usize{ 2, 2 };
    const strides = [_]usize{ 1, 1 };
    const dilations = [_]usize{ 1, 1 };
    const pads = [_]usize{ 0, 0, 0, 0 };

    try TensMath.onnx_averagepool_lean(T, &input, &output, &kernel_shape, &strides, &dilations, &pads, .NOTSET, false);

    const expected = [_]f32{ 3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5 };
    for (output.data, expected) |got, exp| {
        try std.testing.expectApproxEqAbs(got, exp, 1e-5);
    }
}
