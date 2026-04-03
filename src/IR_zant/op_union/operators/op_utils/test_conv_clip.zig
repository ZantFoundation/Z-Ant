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

const tests_log = std.log.scoped(.test_conv_clip);

// ---------------------------------------------------------------
// -------------------- TESTS FOR CONV+CLIP ----------------------
// ---------------------------------------------------------------
// Conv+Clip tests for convolution followed by clipping operation

// Test clipping values above max
test "Conv_Clip - values clipped to max" {
    tests_log.info("\n     test: Conv_Clip - values clipped to max\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: all 1s
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // Kernel: all 1s (will produce value 4)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Min and Max tensors
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{-10.0};
    var maxArray: [1]f32 = [_]f32{2.0}; // Clip to max 2.0

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    // Create output tensor with correct shape [1, 1, 2, 2]
    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[3]);

    // Without clip would be 4, with clip becomes 2.0 (max)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 2.0), val);
    }

    tests_log.info("Values correctly clipped to max\n", .{});
}

// Test clipping values below min
test "Conv_Clip - values clipped to min" {
    tests_log.info("\n     test: Conv_Clip - values clipped to min\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: all 1s
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // Kernel: all -1s (will produce value -4)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1, -1 },
                [_]f32{ -1, -1 },
            },
        },
    };

    // Min and Max tensors
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{-2.0}; // Clip to min -2.0
    var maxArray: [1]f32 = [_]f32{10.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    // Create output tensor with correct shape [1, 1, 2, 2]
    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Without clip would be -4, with clip becomes -2.0 (min)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, -2.0), val);
    }

    tests_log.info("Values correctly clipped to min\n", .{});
}

// Test values within clip range (no clipping needed)
test "Conv_Clip - values within range unchanged" {
    tests_log.info("\n     test: Conv_Clip - values within range unchanged\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: all 1s
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // Kernel: all 1s (will produce value 4)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Min and Max tensors with wide range
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{10.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Value 4 is within [0, 10] range
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 4), val);
    }

    tests_log.info("Values within range preserved correctly\n", .{});
}

// Test mixed values with different clipping behaviors
test "Conv_Clip - mixed values with selective clipping" {
    tests_log.info("\n     test: Conv_Clip - mixed values with selective clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input with variable values
    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 1, -1, 2, 1 },
                [_]f32{ 0, 3, -2, 1 },
                [_]f32{ 2, 1, 4, 0 },
                [_]f32{ -1, 0, 1, 2 },
            },
        },
    };

    // Kernel: mix of positive and negative
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 0.5 },
                [_]f32{ 0.5, 1 },
            },
        },
    };

    // Clip range [-2, 3]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{-2.0};
    var maxArray: [1]f32 = [_]f32{3.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Output shape: [1, 1, 3, 3]
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]);

    // All values should be within [-2, 3]
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= -2.0);
        try std.testing.expect(val <= 3.0);
    }

    tests_log.info("Mixed values clipped correctly\n", .{});
}

// Test with bias and clipping
test "Conv_Clip - with bias affecting clip behavior" {
    tests_log.info("\n     test: Conv_Clip - with bias affecting clip behavior\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Bias: 3 (conv result 4 + bias 3 = 7, will be clipped to 5)
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{3};

    // Clip range [0, 5]
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{5.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &bias_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &bias_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, &bias_tensor, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Result should be 5.0 (7.0 clipped to max 5.0)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5.0), val);
    }

    tests_log.info("Bias with clipping handled correctly\n", .{});
}

// Test with only max clipping (no min)
test "Conv_Clip - with only max bound" {
    tests_log.info("\n     test: Conv_Clip - with only max bound\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Only max, no min
    var max_shape: [1]usize = [_]usize{1};
    var maxArray: [1]f32 = [_]f32{2.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &max_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, null, &max_tensor);

    // Without clip would be 4, with max=2 becomes 2
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 2.0), val);
    }

    tests_log.info("Max-only clipping works correctly\n", .{});
}

// Test multi-channel with clipping
test "Conv_Clip - multi-channel with clipping" {
    tests_log.info("\n     test: Conv_Clip - multi-channel with clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    // 1 batch, 2 input channels, 3x3
    var input_shape: [4]usize = [_]usize{ 1, 2, 3, 3 };
    var inputArray: [1][2][3][3]f32 = [_][2][3][3]f32{
        [_][3][3]f32{
            // Channel 1
            [_][3]f32{
                [_]f32{ 2, 3, 2 },
                [_]f32{ 1, 4, 1 },
                [_]f32{ 2, 3, 2 },
            },
            // Channel 2
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // 1 filter, 2 input channels, 2x2 kernel
    var kernel_shape: [4]usize = [_]usize{ 1, 2, 2, 2 };
    var kernelArray: [1][2][2][2]f32 = [_][2][2][2]f32{
        [_][2][2]f32{
            // Channel 1
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
            // Channel 2
            [_][2]f32{
                [_]f32{ 0.5, 0.5 },
                [_]f32{ 0.5, 0.5 },
            },
        },
    };

    // Clip range [0, 10]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{10.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[3]);

    // All values should be within [0, 10]
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 10.0);
    }

    tests_log.info("Multi-channel with clipping works correctly\n", .{});
}

// Test with padding and clipping
test "Conv_Clip - SAME_UPPER padding with clipping" {
    tests_log.info("\n     test: Conv_Clip - SAME_UPPER padding with clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 2, 2, 2, 2 },
                [_]f32{ 2, 2, 2, 2 },
                [_]f32{ 2, 2, 2, 2 },
                [_]f32{ 2, 2, 2, 2 },
            },
        },
    };

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

    // Clip range [0, 12]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{12.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const auto_pad = "SAME_UPPER";

    var output_shape = [_]usize{ 1, 1, 4, 4 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, null, null, null, auto_pad, &min_tensor, &max_tensor);

    try std.testing.expectEqual(@as(usize, 4), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), output_tensor.shape[3]);

    // All values should be within [0, 12]
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 12.0);
    }

    tests_log.info("SAME_UPPER padding with clipping works correctly\n", .{});
}

// Test with stride, dilation, and clipping
test "Conv_Clip - stride and dilation with clipping" {
    tests_log.info("\n     test: Conv_Clip - stride and dilation with clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 6, 6 };
    var inputArray: [1][1][6][6]f32 = [_][1][6][6]f32{
        [_][6][6]f32{
            [_][6]f32{
                [_]f32{ 1, 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Clip range [0, 3]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{3.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{2};
    const dilations = [_]usize{2};

    // Output con stride=2, dilation=2: [1, 1, 2, 2]
    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, null, &dilations, null, null, &min_tensor, &max_tensor);

    // All values should be within [0, 3] and clipped to 3 (result would be 4 without clip)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 3.0), val);
    }

    tests_log.info("Stride and dilation with clipping works correctly\n", .{});
}

// Test depthwise convolution with clipping
test "Conv_Clip - depthwise with clipping" {
    tests_log.info("\n     test: Conv_Clip - depthwise with clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    // 1 batch, 2 input channels, 3x3
    var input_shape: [4]usize = [_]usize{ 1, 2, 3, 3 };
    var inputArray: [1][2][3][3]f32 = [_][2][3][3]f32{
        [_][3][3]f32{
            // Channel 1
            [_][3]f32{
                [_]f32{ 2, 2, 2 },
                [_]f32{ 2, 2, 2 },
                [_]f32{ 2, 2, 2 },
            },
            // Channel 2
            [_][3]f32{
                [_]f32{ 3, 3, 3 },
                [_]f32{ 3, 3, 3 },
                [_]f32{ 3, 3, 3 },
            },
        },
    };

    // 2 filters, 1 channel each (depthwise)
    var kernel_shape: [4]usize = [_]usize{ 2, 1, 2, 2 };
    var kernelArray: [2][1][2][2]f32 = [_][1][2][2]f32{
        // Filter 1
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
        // Filter 2
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Clip range [0, 10]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{10.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 2, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, 2, null, &min_tensor, &max_tensor);

    // Output shape: [1, 2, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[3]);

    // All values should be within [0, 10]
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 10.0);
    }

    tests_log.info("Depthwise with clipping works correctly\n", .{});
}

// Test multi-batch with clipping
test "Conv_Clip - multi-batch with clipping" {
    tests_log.info("\n     test: Conv_Clip - multi-batch with clipping\n", .{});

    const allocator = pkgAllocator.allocator;

    // 2 batches, 1 channel, 3x3
    var input_shape: [4]usize = [_]usize{ 2, 1, 3, 3 };
    var inputArray: [2][1][3][3]f32 = [_][1][3][3]f32{
        // Batch 1
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 3, 3, 3 },
                [_]f32{ 3, 3, 3 },
                [_]f32{ 3, 3, 3 },
            },
        },
        // Batch 2
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 2, 2, 2 },
                [_]f32{ 2, 2, 2 },
                [_]f32{ 2, 2, 2 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Clip range [0, 10]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{10.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 2, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // Output shape: [2, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output_tensor.shape[3]);

    // Batch 1: 3*4 = 12, clipped to 10
    // Batch 2: 2*4 = 8, within range
    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, 10.0), output_tensor.data[i]);
    }
    for (4..8) |i| {
        try std.testing.expectEqual(@as(f32, 8.0), output_tensor.data[i]);
    }

    tests_log.info("Multi-batch with clipping works correctly\n", .{});
}

// Test extreme clipping (min = max)
test "Conv_Clip - extreme clipping min equals max" {
    tests_log.info("\n     test: Conv_Clip - extreme clipping min equals max\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
                [_]f32{ 7, 8, 9 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0.1, 0.2 },
                [_]f32{ 0.3, 0.4 },
            },
        },
    };

    // Min = Max = 5.0 (all values become 5.0)
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{5.0};
    var maxArray: [1]f32 = [_]f32{5.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // All values should be exactly 5.0
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5.0), val);
    }

    tests_log.info("Extreme clipping (min=max) works correctly\n", .{});
}

// Test with conv_clip_lean
test "Conv_Clip - conv_clip_lean with bias and dilation" {
    tests_log.info("\n     test: Conv_Clip - conv_clip_lean with bias and dilation\n", .{});

    const allocator = pkgAllocator.allocator;

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

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{1};

    // Clip range [0, 4]
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{4.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &bias_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &bias_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const dilations = [_]usize{2};

    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, &bias_tensor, &stride, null, &dilations, null, null, &min_tensor, &max_tensor);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]);

    // Each output: 4 (dilated kernel) + 1 (bias) = 5, clipped to 4
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 4.0), val);
    }

    tests_log.info("Conv_clip_lean with bias, dilation and clipping works correctly\n", .{});
}

// Test zero-centered clipping (symmetric range)
test "Conv_Clip - symmetric clipping range" {
    tests_log.info("\n     test: Conv_Clip - symmetric clipping range\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ -2, 3, -1, 2 },
                [_]f32{ 1, -3, 2, -1 },
                [_]f32{ -1, 2, 3, 1 },
                [_]f32{ 2, -1, -2, 3 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0.5, -0.5 },
                [_]f32{ -0.5, 0.5 },
            },
        },
    };

    // Symmetric clip range [-3, 3]
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{-3.0};
    var maxArray: [1]f32 = [_]f32{3.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null, &min_tensor, &max_tensor);

    // All values should be within [-3, 3]
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= -3.0);
        try std.testing.expect(val <= 3.0);
    }

    tests_log.info("Symmetric clipping range works correctly\n", .{});
}

// ===============================================================
// -------------------- EDGE CASES TESTS -------------------------
// ===============================================================

// Edge case: ReLU6 pattern (Clip(0, 6)) - Common in MobileNet v2
test "Conv_Clip - ReLU6 pattern (MobileNet v2)" {
    tests_log.info("\n     test: Conv_Clip - ReLU6 pattern (MobileNet v2)\n", .{});

    const allocator = pkgAllocator.allocator;

    // Simulating a typical depthwise separable convolution output
    var input_shape: [4]usize = [_]usize{ 1, 3, 4, 4 };
    var inputArray: [1][3][4][4]f32 = undefined;

    // Fill with varied values to test ReLU6 behavior
    for (0..3) |c| {
        for (0..4) |h| {
            for (0..4) |w| {
                const val = @as(f32, @floatFromInt(c * 16 + h * 4 + w)) * 0.5 - 5.0;
                inputArray[0][c][h][w] = val;
            }
        }
    }

    // 3 filters, 1 channel each (depthwise)
    var kernel_shape: [4]usize = [_]usize{ 3, 1, 3, 3 };
    var kernelArray: [3][1][3][3]f32 = undefined;

    for (0..3) |f| {
        for (0..3) |h| {
            for (0..3) |w| {
                kernelArray[f][0][h][w] = 0.1;
            }
        }
    }

    // ReLU6: Clip(0, 6)
    var min_shape: [1]usize = [_]usize{1};
    var minArray: [1]f32 = [_]f32{0.0};
    var maxArray: [1]f32 = [_]f32{6.0};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var min_tensor = try Tensor(f32).fromArray(&allocator, &minArray, &min_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &maxArray, &min_shape);
    defer max_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var output_shape = [_]usize{ 1, 3, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_clip_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, 3, null, &min_tensor, &max_tensor);

    // Verify all values are in [0, 6] range (ReLU6)
    for (output_tensor.data) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 6.0);
    }

    tests_log.info("ReLU6 pattern (Clip(0,6)) works correctly\n", .{});
}
