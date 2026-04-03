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

const tests_log = std.log.scoped(.test_conv_relu);

// ---------------------------------------------------------------
// -------------------- TESTs FOR CONV+RELU ----------------------
// ---------------------------------------------------------------
// Conv+ReLU has tests for convolution + tests for the ReLU operation

// Verifica che ReLU azzeri i valori negativi
test "Conv_ReLU - negative values become zero" {
    tests_log.info("\n     test: Conv_ReLU - negative values become zero\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: tutti 1
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

    // Kernel: tutti -1 (produrrà valori negativi)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1, -1 },
                [_]f32{ -1, -1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[3]);

    // Senza ReLU sarebbe -4, con ReLU diventa 0
    for (result.data) |val| {
        try std.testing.expectEqual(@as(f32, 0), val);
    }

    tests_log.info("All negative values correctly clipped to zero\n", .{});
}

//Tutti valori positivi (ReLU non fa nulla)
test "Conv_ReLU - all positive values unchanged" {
    tests_log.info("\n     test: Conv_ReLU - all positive values unchanged\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: tutti 1
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

    // Kernel: tutti 1 (produrrà valori positivi)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Ogni output è 4 (somma di 2x2 kernel di 1)
    for (result.data) |val| {
        try std.testing.expectEqual(@as(f32, 4), val);
    }

    tests_log.info("Positive values preserved correctly\n", .{});
}

// Tutti valori a 0 (ReLU non fa nulla)
test "Conv_ReLU - all 0 values" {
    tests_log.info("\n     test: Conv_ReLU - all 0 values\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input con valori variabili
    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 1, 2, 3, 4 },
                [_]f32{ 5, 6, 7, 8 },
                [_]f32{ 9, 10, 11, 12 },
                [_]f32{ 13, 14, 15, 16 },
            },
        },
    };

    // Kernel: mix di positivi e negativi
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, -1 },
                [_]f32{ -1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 3, 3]
    try std.testing.expectEqual(@as(usize, 3), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), result.shape[3]);

    const expected_values = [_]f32{
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };

    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(expected_values[i], val);
    }

    tests_log.info("All 0 values handled correctly\n", .{});
}

// Mix valori positivi e negativi
test "Conv_ReLU - mixed positive and negative values" {
    tests_log.info("\n     test: Conv_ReLU - mixed positive and negative values\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input con valori variabili
    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 1, 1, 5, 8 },
                [_]f32{ 2, 3, 3, 6 },
                [_]f32{ 8, 5, 5, 9 },
                [_]f32{ 1, 1, 3, 2 },
            },
        },
    };

    // Kernel: mix di positivi e negativi
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, -1 },
                [_]f32{ -1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 3, 3]
    try std.testing.expectEqual(@as(usize, 3), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), result.shape[3]);

    // (!) The expected_values have been automatically generated but not verified
    const expected_values = [_]f32{
        1, 0, 0,
        0, 0, 1,
        3, 2, 0,
    };

    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(expected_values[i], val);
    }

    tests_log.info("Mixed values handled correctly\n", .{});
}

// Bias che produce negativi
test "Conv_ReLU - with negative bias" {
    tests_log.info("\n     test: Conv_ReLU - with negative bias\n", .{});

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

    // Bias negativo: 4 (conv) - 10 (bias) = -6 → ReLU: 0
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{-10};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, &pads, null, null, null);
    defer result.deinit();

    // Tutti gli output dovrebbero essere 0 (clipped da ReLU)
    for (result.data) |val| {
        try std.testing.expectEqual(@as(f32, 0), val);
    }

    tests_log.info("Negative bias correctly handled\n", .{});
}

// Bias positivo
test "Conv_ReLU - with positive bias" {
    tests_log.info("\n     test: Conv_ReLU - with positive bias\n", .{});

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

    // Bias positivo: 4 (conv) + 3 (bias) = 7
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{3};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, &pads, null, null, null);
    defer result.deinit();

    // Tutti gli output dovrebbero essere 7
    for (result.data) |val| {
        try std.testing.expectEqual(@as(f32, 7), val);
    }

    tests_log.info("Positive bias correctly added\n", .{});
}

// 1 filtro con 2 canali
test "Conv_ReLU - single filter, two channels" {
    tests_log.info("\n     test: Conv_ReLU - single filter, two channels\n", .{});

    const allocator = pkgAllocator.allocator;

    // 1 batch, 2 canali input, 3x3
    var input_shape: [4]usize = [_]usize{ 1, 2, 3, 3 };
    var inputArray: [1][2][3][3]f32 = [_][2][3][3]f32{
        [_][3][3]f32{
            // Channel 1
            [_][3]f32{
                [_]f32{ 1, 2, 1 },
                [_]f32{ 0, -1, 0 },
                [_]f32{ 2, 1, 2 },
            },
            // Channel 2
            [_][3]f32{
                [_]f32{ -1, -1, -1 },
                [_]f32{ 0, 1, 0 },
                [_]f32{ 1, 0, -1 },
            },
        },
    };

    // 1 filtro, 2 canali input, kernel 2x2
    var kernel_shape: [4]usize = [_]usize{ 1, 2, 2, 2 };
    var kernelArray: [1][2][2][2]f32 = [_][2][2][2]f32{
        [_][2][2]f32{
            // Channel 1
            [_][2]f32{
                [_]f32{ 1, -1 },
                [_]f32{ 0, 1 },
            },
            // Channel 2
            [_][2]f32{
                [_]f32{ -1, 0 },
                [_]f32{ 1, -1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[3]);

    // (!) expected results have been automatically generated but not verified
    const expected = [_]f32{
        0, 3, // -2 -> 0
        3, 1,
    };

    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(expected[i], val);
    }

    tests_log.info("Single filter, 2 channels works correctly \n", .{});
}

// Depthwise convolution: 2 filtri, 1 canale ciascuno
test "Conv_ReLU - depthwise two filters" {
    tests_log.info("\n     test: Conv_ReLU - depthwise two filters\n", .{});

    const allocator = pkgAllocator.allocator;

    // 1 batch, 2 canali input, 3x3
    var input_shape: [4]usize = [_]usize{ 1, 2, 3, 3 };
    var inputArray: [1][2][3][3]f32 = [_][2][3][3]f32{
        [_][3][3]f32{
            // Channel 1
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
                [_]f32{ 7, 8, 9 },
            },
            // Channel 2
            [_][3]f32{
                [_]f32{ 9, 8, 7 },
                [_]f32{ 6, 5, 4 },
                [_]f32{ 3, 2, 1 },
            },
        },
    };

    // 2 filtri, 1 canale ciascuno
    var kernel_shape: [4]usize = [_]usize{ 2, 1, 2, 2 };
    var kernelArray: [2][1][2][2]f32 = [_][1][2][2]f32{
        // Filtro 1 → agisce solo sul canale 1
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, -1 },
                [_]f32{ 0, 1 },
            },
        },
        // Filtro 2 → agisce solo sul canale 2
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1, 1 },
                [_]f32{ 0, -1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, 2, null);
    defer result.deinit();

    // Output shape: [1, 2, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[3]);

    // (!) expected results have been automatically generated but not verified
    const expected: [1][2][2][2]f32 = [_][2][2][2]f32{[_][2][2]f32{
        [_][2]f32{
            [_]f32{ 4, 5 },
            [_]f32{ 7, 8 },
        },
        [_][2]f32{
            [_]f32{ 0, 0 }, // -6 -> 0, -5 -> 0
            [_]f32{ 0, 0 }, // -3 -> 0, -2 -> 0
        },
    }};

    var idx: usize = 0;
    for (expected) |batch| {
        for (batch) |channel| {
            for (channel) |row| {
                for (row) |e| {
                    try std.testing.expectEqual(@as(f32, e), result.data[idx]);
                    idx += 1;
                }
            }
        }
    }

    tests_log.info("Depthwise two filters works correctly\n", .{});
}

// Multi-batch, multi-channel, multi-filter
// TODO: replace expected_results with correct values
test "Conv_ReLU - multi batch, channel and filter" {
    tests_log.info("\n     test: Conv_ReLU - multi batch, channel and filter\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 1.0, -2.0, 1.0 },
                [_]f32{ -1.0, 0.0, 2.0 },
                [_]f32{ 1.0, -1.0, 1.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 0.0, 1.0, -1.0 },
                [_]f32{ -2.0, 1.0, 0.0 },
                [_]f32{ 1.0, 0.0, -1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ -1.0, 2.0, -2.0 },
                [_]f32{ 1.0, -1.0, 0.0 },
                [_]f32{ 0.0, 1.0, -1.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 2.0, -1.0, 1.0 },
                [_]f32{ -1.0, 0.0, -2.0 },
                [_]f32{ 1.0, -1.0, 0.0 },
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
                [_]f32{ -1.0, 1.0 },
                [_]f32{ 1.0, -1.0 },
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
                [_]f32{ 0.5, -0.5 },
                [_]f32{ -0.5, 0.5 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ -0.5, 0.5 },
                [_]f32{ 0.5, -0.5 },
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

    var result_tensor = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, &bias, &stride, null, null, null, null);
    defer result_tensor.deinit();

    // Expected results with the correct dimensions
    // (!) expected results have been automatically generated but not verified
    const expected_result: [2][2][2][2]f32 = [_][2][2][2]f32{
        // First batch
        [_][2][2]f32{
            // First filter
            [_][2]f32{
                [_]f32{ 0.0, 3.0 }, // -1 -> 0
                [_]f32{ 0.0, 1.0 },
            },
            // Second filter
            [_][2]f32{
                [_]f32{ 2.0, 0.0 }, // -1 -> 0
                [_]f32{ 1.5, 1.0 },
            },
        },
        // Second batch
        [_][2][2]f32{
            // First filter
            [_][2]f32{
                [_]f32{ 10.0, 0.0 }, // -8 -> 0
                [_]f32{ 0.0, 7.0 }, // -5 -> 0
            },
            // Second filter
            [_][2]f32{
                [_]f32{ 0.0, 5.5 }, // -3.5 -> 0
                [_]f32{ 4.0, 0.0 }, // -2 -> 0
            },
        },
    };

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
                    try std.testing.expectEqual(expected_result[batch][filter][row][col], result_tensor.get_at(output_location));
                }
            }
        }
    }

    tests_log.info("Multi-batch, channel, filter works correctly\n", .{});
}

// SAME_UPPER padding
test "Conv_ReLU - SAME_UPPER padding" {
    tests_log.info("\n     test: Conv_ReLU - SAME_UPPER padding\n", .{});

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

    var result = try TensMath.conv_relu(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, auto_pad);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 5), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 5), result.shape[3]);

    // Con padding e kernel di 1, i valori sono tutti positivi
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

    tests_log.info("SAME_UPPER padding works correctly\n", .{});
}

// Stride and Padding
test "Conv_Relu - Stride and padding" {
    tests_log.info("\n     test: Conv_Relu - Stride and padding\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 1, 1, 4, 4 };
    var input = try Tensor(f32).fromShape(&allocator, &input_shape);
    defer input.deinit();

    for (0..16) |i| {
        input.data[i] = 1.0;
    }

    var weight_shape = [_]usize{ 1, 1, 3, 3 };
    var weight = try Tensor(f32).fromShape(&allocator, &weight_shape);
    defer weight.deinit();

    for (0..9) |i| {
        weight.data[i] = 1.0;
    }

    const stride_vals = [_]usize{ 2, 2 };
    const pad_vals = [_]usize{ 1, 1, 1, 1 };

    var result = try TensMath.conv_relu(f32, &input, &weight, null, &stride_vals, &pad_vals, null, null, null);
    defer result.deinit();

    // With stride=2 and padding=1, output should be [1, 1, 2, 2]
    try std.testing.expect(result.shape[2] == 2);
    try std.testing.expect(result.shape[3] == 2);

    tests_log.info("Stride and padding works correctly\n", .{});
}

// Bias and Dilation
test "Conv_Relu - conv_relu_lean with bias and dilation" {
    tests_log.info("\n     test: Conv_Relu - conv_relu_lean with bias and dilation\n", .{});

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

    try TensMath.conv_relu_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, &bias_tensor, &stride, null, &dilations, null, null);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]); // width

    // Each output should be 5 (4 from dilated kernel + 1 from bias)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5), val);
    }

    tests_log.info("Conv_relu_lean with bias and dilation works correctly\n", .{});
}
