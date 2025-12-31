const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const quant = zant.core.quantization;
//import error library
const TensorError = zant.utils.error_handler.TensorError;
const pkgAllocator = zant.utils.allocator;

const from_NHWC_to_NCHW = zant.core.tensor.from_NHWC_to_NCHW;
const from_NCHW_to_NHWC = zant.core.tensor.from_NCHW_to_NHWC;

const tests_log = std.log.scoped(.test_lib_shape);

const expect = std.testing.expect;

test {
    _ = @import("TensorMath/test_tensor_math.zig");
}

test "Tensor test description" {
    tests_log.info("\n--- Running tensor tests\n", .{});
}

test "init() test" {
    tests_log.info("\n     test: init() ", .{});
    const allocator = pkgAllocator.allocator;
    var tensor = try Tensor(f64).init(&allocator);
    defer tensor.deinit();
    const size = tensor.getSize();
    try std.testing.expect(size == 0);
    try std.testing.expect(&allocator == tensor.allocator);
}

test "initialization fromShape" {
    tests_log.info("\n     test:initialization fromShape", .{});
    const allocator = pkgAllocator.allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).fromShape(&allocator, &shape);
    defer tensor.deinit();
    const size = tensor.getSize();
    try std.testing.expect(size == 6);
    for (0..tensor.size) |i| {
        const val = try tensor.get(i);
        try std.testing.expect(val == 0);
    }
}

test "Get_Set_Test" {
    tests_log.info("\n     test:Get_Set_Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.set(5, 33);
    const val = try tensor.get(5);

    try std.testing.expect(val == 33);
}

test "Flatten Index Test" {
    tests_log.info("\n     test:Flatten Index Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var indices = [_]usize{ 1, 2 };
    const flatIndex = try tensor.flatten_index(&indices);

    //tests_log.info("\nflatIndex: {}\n", .{flatIndex});
    try std.testing.expect(flatIndex == 5);
    indices = [_]usize{ 0, 0 };
    const flatIndex2 = try tensor.flatten_index(&indices);
    //tests_log.info("\nflatIndex2: {}\n", .{flatIndex2});
    try std.testing.expect(flatIndex2 == 0);
}

test "Get_at Set_at Test" {
    tests_log.info("\n     test:Get_at Set_at Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var indices = [_]usize{ 1, 1 };
    var value = try tensor.get_at(&indices);
    try std.testing.expect(value == 5.0);

    for (0..2) |i| {
        for (0..3) |j| {
            indices[0] = i;
            indices[1] = j;
            value = try tensor.get_at(&indices);
            try std.testing.expect(value == i * 3 + j + 1);
        }
    }

    try tensor.set_at(&indices, 1.0);
    value = try tensor.get_at(&indices);
    try std.testing.expect(value == 1.0);
}

test " copy() method" {
    tests_log.info("\n     test:copy() method ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 20, 30 },
        [_]u8{ 40, 50, 60 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensorCopy = try tensor.copy();
    defer tensorCopy.deinit();

    for (0..tensor.data.len) |i| {
        try std.testing.expect(tensor.data[i] == tensorCopy.data[i]);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expect(tensor.shape[i] == tensorCopy.shape[i]);
    }
}

test "to array " {
    tests_log.info("\n     test:to array ", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    const array_from_tensor = try tensor.toArray(shape.len);
    defer allocator.free(array_from_tensor);

    try expect(array_from_tensor.len == 2);
    try expect(array_from_tensor[0].len == 3);
}

test "test setToZero() " {
    tests_log.info("\n     test: setToZero()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]u8 = [_][3][3]u8{
        [_][3]u8{
            [_]u8{ 10, 20, 30 },
            [_]u8{ 40, 50, 60 },
            [_]u8{ 70, 80, 90 },
        },
        [_][3]u8{
            [_]u8{ 10, 20, 30 },
            [_]u8{ 40, 50, 60 },
            [_]u8{ 70, 80, 90 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.setToZero();

    for (tensor.data) |d| {
        try std.testing.expectEqual(d, 0);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expectEqual(tensor.shape[i], shape[i]);
    }
}

test "ensure_4D_shape" {
    tests_log.info("\n     test: ensure_4D_shape ", .{});

    //shape 1D
    const shape = [_]usize{5};
    var result = try Tensor(f32).ensure_4D_shape(&shape);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 1);
    try std.testing.expectEqual(result[3], 5);

    //shape 2D
    const shape_2 = [_]usize{ 5, 10 };
    result = try Tensor(f32).ensure_4D_shape(&shape_2);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 5);
    try std.testing.expectEqual(result[3], 10);

    //shape 3D
    tests_log.info("\n     test: ensure_4D_shape with 3 dimensions", .{});

    const shape_3 = [_]usize{ 5, 10, 15 };
    result = try Tensor(f32).ensure_4D_shape(&shape_3);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 5);
    try std.testing.expectEqual(result[2], 10);
    try std.testing.expectEqual(result[3], 15);

    //shape 4D
    tests_log.info("\n     test: ensure_4D_shape with 4 dimensions", .{});

    const shape_4 = [_]usize{ 5, 10, 15, 20 };
    result = try Tensor(f32).ensure_4D_shape(&shape_4);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 5);
    try std.testing.expectEqual(result[1], 10);
    try std.testing.expectEqual(result[2], 15);
    try std.testing.expectEqual(result[3], 20);

    // shape 5D --> check for error
    tests_log.info("\n     test: ensure_4D_shape with 5 dimensions", .{});

    const shape_5 = [_]usize{ 5, 10, 15, 20, 25 };

    try std.testing.expectError(error.InvalidDimensions, Tensor(f32).ensure_4D_shape(&shape_5));
}

test "benchmark flatten_index implementations" {
    tests_log.info("\n     test: benchmark flatten_index implementations", .{});
    const allocator = pkgAllocator.allocator;

    // Test with different tensor dimensions
    const iterations = 1_000_000;
    const benchmark_runs = 10; // Number of times to run each benchmark for averaging

    // 1D tensor benchmark
    {
        var shape_1d = [_]usize{100};
        var tensor_1d = try Tensor(f32).fromShape(&allocator, &shape_1d);
        defer tensor_1d.deinit();

        var total_optimized: u64 = 0;
        var total_original: u64 = 0;

        for (0..benchmark_runs) |_| {
            const result = tensor_1d.benchmark_flatten_index(iterations);
            total_optimized += result.optimized;
            total_original += result.original;
        }

        const avg_optimized = total_optimized / benchmark_runs;
        const avg_original = total_original / benchmark_runs;
        const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

        tests_log.info("\n       1D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

        // Ensure optimized is at least as fast as original
        try std.testing.expect(avg_optimized <= avg_original + 5);
    }

    // 2D tensor benchmark
    {
        var shape_2d = [_]usize{ 50, 50 };
        var tensor_2d = try Tensor(f32).fromShape(&allocator, &shape_2d);
        defer tensor_2d.deinit();

        var total_optimized: u64 = 0;
        var total_original: u64 = 0;

        for (0..benchmark_runs) |_| {
            const result = tensor_2d.benchmark_flatten_index(iterations);
            total_optimized += result.optimized;
            total_original += result.original;
        }

        const avg_optimized = total_optimized / benchmark_runs;
        const avg_original = total_original / benchmark_runs;
        const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

        tests_log.info("\n       2D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

        try std.testing.expect(avg_optimized <= avg_original);
    }

    // 3D tensor benchmark
    // {
    //     var shape_3d = [_]usize{ 20, 20, 20 };
    //     var tensor_3d = try Tensor(f32).fromShape(&allocator, &shape_3d);
    //     defer tensor_3d.deinit();

    //     var total_optimized: u64 = 0;
    //     var total_original: u64 = 0;

    //     for (0..benchmark_runs) |_| {
    //         const result = tensor_3d.benchmark_flatten_index(iterations);
    //         total_optimized += result.optimized;
    //         total_original += result.original;
    //     }

    //     const avg_optimized = total_optimized / benchmark_runs;
    //     const avg_original = total_original / benchmark_runs;
    //     const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

    //     tests_log.info("\n       3D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

    //     try std.testing.expect(avg_optimized <= avg_original);
    // }

    // // 4D tensor benchmark
    // {
    //     var shape_4d = [_]usize{ 10, 10, 10, 10 };
    //     var tensor_4d = try Tensor(f32).fromShape(&allocator, &shape_4d);
    //     defer tensor_4d.deinit();

    //     var total_optimized: u64 = 0;
    //     var total_original: u64 = 0;

    //     for (0..benchmark_runs) |_| {
    //         const result = tensor_4d.benchmark_flatten_index(iterations);
    //         total_optimized += result.optimized;
    //         total_original += result.original;
    //     }

    //     const avg_optimized = total_optimized / benchmark_runs;
    //     const avg_original = total_original / benchmark_runs;
    //     const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

    //     tests_log.info("\n       4D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

    //     try std.testing.expect(avg_optimized <= avg_original);
    // }

    // 5D tensor benchmark
    // {
    //     var shape_5d = [_]usize{ 5, 5, 5, 5, 5 };
    //     var tensor_5d = try Tensor(f32).fromShape(&allocator, &shape_5d);
    //     defer tensor_5d.deinit();

    //     var total_optimized: u64 = 0;
    //     var total_original: u64 = 0;

    //     for (0..benchmark_runs) |_| {
    //         const result = tensor_5d.benchmark_flatten_index(iterations);
    //         total_optimized += result.optimized;
    //         total_original += result.original;
    //     }

    //     const avg_optimized = total_optimized / benchmark_runs;
    //     const avg_original = total_original / benchmark_runs;
    //     const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

    //     tests_log.info("\n       5D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

    //     try std.testing.expect(avg_optimized <= @as(u64, @intFromFloat(@as(f32, @floatFromInt(avg_original)) * 1.1)));
    // }

    // // 6D tensor benchmark
    // {
    //     var shape_6d = [_]usize{ 4, 4, 4, 4, 4, 4 };
    //     var tensor_6d = try Tensor(f32).fromShape(&allocator, &shape_6d);
    //     defer tensor_6d.deinit();

    //     var total_optimized: u64 = 0;
    //     var total_original: u64 = 0;

    //     for (0..benchmark_runs) |_| {
    //         const result = tensor_6d.benchmark_flatten_index(iterations);
    //         total_optimized += result.optimized;
    //         total_original += result.original;
    //     }

    //     const avg_optimized = total_optimized / benchmark_runs;
    //     const avg_original = total_original / benchmark_runs;
    //     const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

    //     tests_log.info("\n       6D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

    //     try std.testing.expect(avg_optimized <= avg_original + 4);
    // }

    // // 7D tensor benchmark
    // {
    //     var shape_7d = [_]usize{ 3, 3, 3, 3, 3, 3, 3 };
    //     var tensor_7d = try Tensor(f32).fromShape(&allocator, &shape_7d);
    //     defer tensor_7d.deinit();

    //     var total_optimized: u64 = 0;
    //     var total_original: u64 = 0;

    //     for (0..benchmark_runs) |_| {
    //         const result = tensor_7d.benchmark_flatten_index(iterations);
    //         total_optimized += result.optimized;
    //         total_original += result.original;
    //     }

    //     const avg_optimized = total_optimized / benchmark_runs;
    //     const avg_original = total_original / benchmark_runs;
    //     const speedup = @as(f32, @floatFromInt(avg_original)) / @max(1, @as(f32, @floatFromInt(avg_optimized)));

    //     tests_log.info("\n       7D tensor: optimized={d}ms, original={d}ms, speedup={d:.2}x", .{ avg_optimized, avg_original, speedup });

    //     try std.testing.expect(avg_optimized <= avg_original + 4);
    // }
}

test "from_NHWC_to_NCHW and vice versa" {
    const alloc_value = pkgAllocator.allocator;
    const alloc = &alloc_value;

    // basic case (1x2x3x2)
    {
        const N = 1;
        const H = 2;
        const W = 3;
        const C = 2;

        var nhwc_shape = [_]usize{ N, H, W, C };
        var tensor_nhwc = try Tensor(f32).fromShape(alloc, nhwc_shape[0..]);
        defer tensor_nhwc.deinit();

        for (0..12) |i| {
            tensor_nhwc.data[i] = @floatFromInt(i);
        }

        const tensor_nchw = try from_NHWC_to_NCHW(f32, alloc, &tensor_nhwc);
        defer {
            tensor_nchw.deinit();
            alloc.*.destroy(tensor_nchw);
        }

        // Verify shape
        try std.testing.expectEqual(N, tensor_nchw.shape[0]);
        try std.testing.expectEqual(C, tensor_nchw.shape[1]);
        try std.testing.expectEqual(H, tensor_nchw.shape[2]);
        try std.testing.expectEqual(W, tensor_nchw.shape[3]);

        // Verify data reordering
        const expected = [_]f32{ 0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11 };
        for (expected, 0..) |exp, i| {
            try std.testing.expectEqual(exp, tensor_nchw.data[i]);
        }

        // and viceversa...
        const tensor_viceversa: *Tensor(f32) = try from_NCHW_to_NHWC(f32, alloc, tensor_nchw);
        defer {
            tensor_viceversa.deinit();
            alloc.*.destroy(tensor_viceversa);
        }
        // Verify data reordering
        for (0..12) |i| {
            const res: f32 = @floatFromInt(i);
            try std.testing.expectEqual(res, tensor_viceversa.data[i]);
        }
    }

    //larger tensor (2x3x3x3)
    {
        const N = 2;
        const H = 3;
        const W = 3;
        const C = 3;

        var nhwc_shape = [_]usize{ N, H, W, C };
        var tensor_nhwc = try Tensor(f32).fromShape(alloc, nhwc_shape[0..]);
        defer tensor_nhwc.deinit();

        for (0..tensor_nhwc.data.len) |i| {
            tensor_nhwc.data[i] = @floatFromInt(i);
        }

        const tensor_nchw = try from_NHWC_to_NCHW(f32, alloc, &tensor_nhwc);
        defer {
            tensor_nchw.deinit();
            alloc.*.destroy(tensor_nchw);
        }

        // Verify shape
        try std.testing.expectEqual(N, tensor_nchw.shape[0]);
        try std.testing.expectEqual(C, tensor_nchw.shape[1]);
        try std.testing.expectEqual(H, tensor_nchw.shape[2]);
        try std.testing.expectEqual(W, tensor_nchw.shape[3]);

        // Verify first element of each channel
        try std.testing.expectEqual(@as(f32, 0), tensor_nchw.data[0]); // ch0, pos 0,0
        try std.testing.expectEqual(@as(f32, 1), tensor_nchw.data[9]); // ch1, pos 0,0
        try std.testing.expectEqual(@as(f32, 2), tensor_nchw.data[18]); // ch2, pos 0,0

        // and viceversa...
        const tensor_viceversa = try from_NCHW_to_NHWC(f32, alloc, tensor_nchw);
        defer {
            tensor_viceversa.deinit();
            alloc.*.destroy(tensor_viceversa);
        }
        // Verify data reordering
        for (0..tensor_nhwc.data.len) |i| {
            const res: f32 = @floatFromInt(i);
            try std.testing.expectEqual(res, tensor_viceversa.data[i]);
        }
    }

    //empty tensor
    {
        var nhwc_shape = [_]usize{ 0, 2, 2, 3 };
        var tensor_nhwc = try Tensor(f32).fromShape(alloc, nhwc_shape[0..]);
        defer tensor_nhwc.deinit();

        // Empty the data
        //alloc.*.free(tensor_nhwc.data);
        //tensor_nhwc.data = &[_]f32{};

        const result1 = from_NHWC_to_NCHW(f32, alloc, &tensor_nhwc);
        try std.testing.expectError(error.EmptyTensor, result1);

        const result2 = from_NCHW_to_NHWC(f32, alloc, &tensor_nhwc);
        try std.testing.expectError(error.EmptyTensor, result2);
    }

    //invalid shape (3D tensor should fail)
    {
        var shape_3d = [_]usize{ 2, 3, 4 };
        var tensor_3d = try Tensor(f32).fromShape(alloc, shape_3d[0..]);
        defer tensor_3d.deinit();

        const result = from_NHWC_to_NCHW(f32, alloc, &tensor_3d);
        try std.testing.expectError(error.InvalidShape, result);

        // and viceversa...
        const tensor_viceversa = from_NCHW_to_NHWC(f32, alloc, &tensor_3d);
        try std.testing.expectError(error.InvalidShape, tensor_viceversa);
    }

    //Different type (i32)
    {
        var nhwc_shape = [_]usize{ 1, 2, 2, 2 };
        var tensor_nhwc = try Tensor(i32).fromShape(alloc, nhwc_shape[0..]);
        defer tensor_nhwc.deinit();

        for (0..8) |i| {
            tensor_nhwc.data[i] = @intCast(i);
        }

        const tensor_nchw = try from_NHWC_to_NCHW(i32, alloc, &tensor_nhwc);
        defer {
            tensor_nchw.deinit();
            alloc.*.destroy(tensor_nchw);
        }

        // NHWC: [0,1, 2,3, 4,5, 6,7]
        // NCHW: [0,2,4,6, 1,3,5,7]
        const expected = [_]i32{ 0, 2, 4, 6, 1, 3, 5, 7 };
        for (expected, 0..) |exp, i| {
            try std.testing.expectEqual(exp, tensor_nchw.data[i]);
        }

        // and viceversa...
        const tensor_viceversa = try from_NCHW_to_NHWC(i32, alloc, tensor_nchw);
        defer {
            tensor_viceversa.deinit();
            alloc.*.destroy(tensor_viceversa);
        }
        // Verify data reordering
        var i: i32 = 0;
        for (0..tensor_nhwc.data.len) |j| {
            try std.testing.expectEqual(i, tensor_viceversa.data[j]);
            i += 1;
        }
    }

    //realistic tensor (32x224x224x3)
    {
        const N = 32;
        const H = 224;
        const W = 224;
        const C = 3;

        const total_elements = N * H * W * C;

        var nhwc_shape = [_]usize{ N, H, W, C };
        var tensor_nhwc = try Tensor(f32).fromShape(alloc, nhwc_shape[0..]);
        defer tensor_nhwc.deinit();

        // Fill with deterministic pattern for verification
        for (0..total_elements) |i| {
            tensor_nhwc.data[i] = @floatFromInt(i); // Simulate pixel values 0-255
        }

        const tensor_nchw = try from_NHWC_to_NCHW(f32, alloc, &tensor_nhwc);
        defer {
            tensor_nchw.deinit();
            alloc.*.destroy(tensor_nchw);
        }

        // verify shape
        try std.testing.expectEqual(N, tensor_nchw.shape[0]);
        try std.testing.expectEqual(C, tensor_nchw.shape[1]);
        try std.testing.expectEqual(H, tensor_nchw.shape[2]);
        try std.testing.expectEqual(W, tensor_nchw.shape[3]);

        // verify size
        try std.testing.expectEqual(total_elements, tensor_nchw.data.len);

        // Verify specific elements at key positions

        // First pixel of first image: NHWC[0,0,0,:] -> NCHW[0,:,0,0]
        const first_pixel_r = tensor_nhwc.data[0]; // R channel
        const first_pixel_g = tensor_nhwc.data[1]; // G channel
        const first_pixel_b = tensor_nhwc.data[2]; // B channel

        const pixels_per_channel = H * W;
        try std.testing.expectEqual(first_pixel_r, tensor_nchw.data[0]); // R at [0,0,0,0]
        try std.testing.expectEqual(first_pixel_g, tensor_nchw.data[pixels_per_channel]); // G at [0,1,0,0]
        try std.testing.expectEqual(first_pixel_b, tensor_nchw.data[2 * pixels_per_channel]); // B at [0,2,0,0]

        //last pixel of first image: NHWC[0,223,223,:] -> NCHW[0,:,223,223]
        const last_pixel_idx_nhwc = (H - 1) * W * C + (W - 1) * C;
        const last_pixel_r = tensor_nhwc.data[last_pixel_idx_nhwc];
        const last_pixel_g = tensor_nhwc.data[last_pixel_idx_nhwc + 1];
        const last_pixel_b = tensor_nhwc.data[last_pixel_idx_nhwc + 2];

        const last_pixel_idx_nchw = pixels_per_channel - 1;
        try std.testing.expectEqual(last_pixel_r, tensor_nchw.data[last_pixel_idx_nchw]);
        try std.testing.expectEqual(last_pixel_g, tensor_nchw.data[pixels_per_channel + last_pixel_idx_nchw]);
        try std.testing.expectEqual(last_pixel_b, tensor_nchw.data[2 * pixels_per_channel + last_pixel_idx_nchw]);

        // first pixel of second image: NHWC[1,0,0,:] -> NCHW[1,:,0,0]
        const second_image_offset_nhwc = H * W * C;
        const second_image_r = tensor_nhwc.data[second_image_offset_nhwc];

        const second_image_offset_nchw = C * H * W;
        try std.testing.expectEqual(second_image_r, tensor_nchw.data[second_image_offset_nchw]);

        // middle pixel of middle image for good measure
        const mid_batch = N / 2;
        const mid_h = H / 2;
        const mid_w = W / 2;

        const mid_idx_nhwc = mid_batch * (H * W * C) + mid_h * (W * C) + mid_w * C;
        const mid_pixel_r = tensor_nhwc.data[mid_idx_nhwc];

        const mid_idx_nchw = mid_batch * (C * H * W) + mid_h * W + mid_w;
        try std.testing.expectEqual(mid_pixel_r, tensor_nchw.data[mid_idx_nchw]);

        // and viceversa...
        const tensor_viceversa = try from_NCHW_to_NHWC(f32, alloc, tensor_nchw);
        defer {
            tensor_viceversa.deinit();
            alloc.*.destroy(tensor_viceversa);
        }

        //verify shape
        try std.testing.expectEqual(N, tensor_viceversa.shape[0]);
        try std.testing.expectEqual(H, tensor_viceversa.shape[1]);
        try std.testing.expectEqual(W, tensor_viceversa.shape[2]);
        try std.testing.expectEqual(C, tensor_viceversa.shape[3]);

        //verify data reordering
        for (0..total_elements) |i| {
            try std.testing.expectEqual(tensor_nhwc.data[i], tensor_viceversa.data[i]);
        }
    }
}
