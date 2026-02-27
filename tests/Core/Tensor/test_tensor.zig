const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const quant = zant.core.quantization;
//import error library
const TensorError = zant.utils.error_handler.TensorError;
const pkgAllocator = zant.utils.allocator;

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
    var out: [4]usize = undefined;
    var result = try Tensor(f32).ensure_4D_shape(&shape, &out);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 1);
    try std.testing.expectEqual(result[3], 5);

    //shape 2D
    const shape_2 = [_]usize{ 5, 10 };
    result = try Tensor(f32).ensure_4D_shape(&shape_2, &out);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 5);
    try std.testing.expectEqual(result[3], 10);

    //shape 3D
    tests_log.info("\n     test: ensure_4D_shape with 3 dimensions", .{});

    const shape_3 = [_]usize{ 5, 10, 15 };
    result = try Tensor(f32).ensure_4D_shape(&shape_3, &out);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 5);
    try std.testing.expectEqual(result[2], 10);
    try std.testing.expectEqual(result[3], 15);

    //shape 4D
    tests_log.info("\n     test: ensure_4D_shape with 4 dimensions", .{});

    const shape_4 = [_]usize{ 5, 10, 15, 20 };
    result = try Tensor(f32).ensure_4D_shape(&shape_4, &out);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 5);
    try std.testing.expectEqual(result[1], 10);
    try std.testing.expectEqual(result[2], 15);
    try std.testing.expectEqual(result[3], 20);

    // shape 5D --> check for error
    tests_log.info("\n     test: ensure_4D_shape with 5 dimensions", .{});

    const shape_5 = [_]usize{ 5, 10, 15, 20, 25 };

    try std.testing.expectError(error.InvalidDimensions, Tensor(f32).ensure_4D_shape(&shape_5, &out));
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
