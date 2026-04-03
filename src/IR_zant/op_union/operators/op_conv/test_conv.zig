const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const lowerConv2d = zant.core.tensor.math_standard.lowerConv2d;

const tests_log = std.log.scoped(.test_conv);

test "Convolution 4D Input with 2x2x2x2 Kernel shape" {
    tests_log.info("\n     test: Convolution 4D Input with 2x2x2x2 Kernel shape\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };

    var inputbias: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2

    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &inputbias, &bias_shape);
    defer bias.deinit();
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    const stride: [2]usize = [_]usize{ 1, 1 };

    var result_tensor = try TensMath.convolve_tensor_with_bias(f32, &input_tensor, &kernel_tensor, &bias, &stride, null, 1);
    defer result_tensor.deinit();

    // Expected results with the correct dimensions
    const expected_result: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 3.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 5.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
    };

    // result_tensor.info();
    // result_tensor.print();

    const output_location = try allocator.alloc(usize, 4); //coordinates in the output space, see test below
    defer allocator.free(output_location);
    @memset(output_location, 0);

    for (0..2) |batch| {
        output_location[0] = batch;
        for (0..2) |filter| {
            output_location[1] = filter;
            for (0..2) |row| {
                output_location[2] = row;
                for (0..2) |col| {
                    output_location[3] = col;
                    //tests_log.info("\n get OUTPUT at:{any}", .{output_location});
                    try std.testing.expectEqual(expected_result[batch][filter][row][col], result_tensor.get_at(output_location));
                }
            }
        }
    }
}

test "OnnxConvLean - NOTSET padding" {
    tests_log.info("\n     test: OnnxConvLean - NOTSET padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    // Create output tensor with correct shape
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]); // width

    // Each output should be 9 (sum of 3x3 kernel of ones)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 9), val);
    }
}

test "OnnxConvLean - SAME_UPPER padding" {
    tests_log.info("\n     test: OnnxConvLean - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const auto_pad = "SAME_UPPER";

    // Create output tensor with correct shape (same as input for SAME_UPPER)
    var output_shape = [_]usize{ 1, 1, 5, 5 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, null, null, null, auto_pad);

    // Add debug prints for padded input
    tests_log.debug("\nKernel values:\n", .{});
    var k_row: usize = 0;
    while (k_row < 3) : (k_row += 1) {
        var k_col: usize = 0;
        while (k_col < 3) : (k_col += 1) {
            const idx = k_row * 3 + k_col;
            tests_log.debug("{d:4.1} ", .{kernel_tensor.data[idx]});
        }
        tests_log.debug("\n", .{});
    }

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 5), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 5), output_tensor.shape[3]); // width

    // Center values should be 9, edge values less due to padding
    const expected_values = [_]f32{
        4, 6, 6, 6, 4,
        6, 9, 9, 9, 6,
        6, 9, 9, 9, 6,
        6, 9, 9, 9, 6,
        4, 6, 6, 6, 4,
    };

    tests_log.debug("\nResult shape: {any}\n", .{output_tensor.shape});
    tests_log.debug("\nActual values:\n", .{});
    var row: usize = 0;
    while (row < 5) : (row += 1) {
        var col: usize = 0;
        while (col < 5) : (col += 1) {
            const idx = row * 5 + col;
            tests_log.debug("{d:4.1} ", .{output_tensor.data[idx]});
        }
        tests_log.debug("\n", .{});
    }

    tests_log.debug("\nExpected values:\n", .{});
    row = 0;
    while (row < 5) : (row += 1) {
        var col: usize = 0;
        while (col < 5) : (col += 1) {
            const idx = row * 5 + col;
            tests_log.debug("{d:4.1} ", .{expected_values[idx]});
        }
        tests_log.debug("\n", .{});
    }

    for (output_tensor.data, 0..) |val, i| {
        if (val != expected_values[i]) {
            tests_log.debug("\nMismatch at index {d}: expected {d}, got {d}\n", .{ i, expected_values[i], val });
        }
        try std.testing.expectEqual(expected_values[i], val);
    }
}

test "OnnxConvLean - with bias and dilation" {
    tests_log.info("\n     test: OnnxConvLean - with bias and dilation\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Bias tensor
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{1};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();

    const stride = [_]usize{1};
    const dilations = [_]usize{2};

    // Create output tensor with correct shape
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, &bias_tensor, &stride, null, &dilations, null, null);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]); // width

    // Each output should be 5 (4 from dilated kernel + 1 from bias)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5), val);
    }
}

test "OnnxConv - all padding modes and features" {
    tests_log.info("\n     test: OnnxConv - all padding modes and features\n", .{});

    const allocator = pkgAllocator.allocator;

    // Test 1: NOTSET padding
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
        var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
            [_][3][3]f32{
                [_][3]f32{
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                },
            },
        };

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();

        const stride = [_]usize{1};
        const pads = [_]usize{ 0, 0, 0, 0 };

        var result = try TensMath.conv(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 3), result.shape[3]); // width

        // Each output should be 9 (sum of 3x3 kernel of ones)
        for (result.data) |val| {
            try std.testing.expectEqual(@as(f32, 9), val);
        }
    }

    // Test 2: SAME_UPPER padding
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
        var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
            [_][3][3]f32{
                [_][3]f32{
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                },
            },
        };

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();

        const stride = [_]usize{1};
        const auto_pad = "SAME_UPPER";

        var result = try TensMath.Conv(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, auto_pad);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 5), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 5), result.shape[3]); // width

        // Center values should be 9, edge values less due to padding
        const expected_values = [_]f32{
            4, 6, 6, 6, 4,
            6, 9, 9, 9, 6,
            6, 9, 9, 9, 6,
            6, 9, 9, 9, 6,
            4, 6, 6, 6, 4,
        };

        for (result.data, 0..) |val, i| {
            try std.testing.expectEqual(expected_values[i], val);
        }
    }

    // Test 3: With bias and dilation
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
        var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
            [_][2][2]f32{
                [_][2]f32{
                    [_]f32{ 1, 1 },
                    [_]f32{ 1, 1 },
                },
            },
        };

        // Bias tensor
        var bias_shape: [1]usize = [_]usize{1};
        var biasArray: [1]f32 = [_]f32{1};

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();
        var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
        defer bias_tensor.deinit();

        const stride = [_]usize{1};
        const dilations = [_]usize{2};

        var result = try TensMath.Conv(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, null, &dilations, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 3), result.shape[3]); // width

        // Each output should be 5 (4 from dilated kernel + 1 from bias)
        for (result.data) |val| {
            try std.testing.expectEqual(@as(f32, 5), val);
        }
    }
}
