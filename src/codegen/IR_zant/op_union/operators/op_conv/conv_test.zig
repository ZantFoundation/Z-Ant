const std = @import("std");
const IR_zant = @import("IR_zant");
const pkgAllocator = IR_zant.pkg_allocator;
const TensMath = IR_zant.core.math_standard;
const Tensor = IR_zant.core.tensor.Tensor;

const tests_log = std.log.scoped(.test_conv);

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

        var result = try TensMath.conv(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, auto_pad);
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

        var result = try TensMath.conv(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, null, &dilations, null, null);
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
