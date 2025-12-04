const std = @import("std");
const zant = @import("zant");
const conv = zant.utils.type_converter;
const init = zant.utils.tensor_initializer;

// Aggiungi questi import per i test NCHW->NHWC:
const IR_zant = @import("IR_zant");
const utils = IR_zant.utils;
const TensorZant = IR_zant.TensorZant;
const TensorCategory = IR_zant.TensorCategory;
const Tensor = zant.core.tensor.Tensor;
const AnyTensor = zant.core.tensor.AnyTensor;
const from_NCHW_to_NHWC = utils.from_NCHW_to_NHWC;

const tests_log = std.log.scoped(.test_utils);

test "Utils description test" {
    tests_log.info("\n--- Running utils test\n", .{});
}

test "convert integer to float" {
    tests_log.info("\n     convert integer to float", .{});
    const result = conv.convert(i32, f64, 42);
    const a: f64 = 42.0;
    try std.testing.expectEqual(a, result);
    try std.testing.expectEqual(f64, @TypeOf(result));
}

test "convert float to integer" {
    tests_log.info("\n     convert float to integer", .{});
    const result = conv.convert(f64, i32, 42.9);
    const a: i32 = 42;
    try std.testing.expectEqual(a, result);
    try std.testing.expectEqual(i32, @TypeOf(result));
}

test "convert integer to bool" {
    tests_log.info("\n     convert integer to bool", .{});
    const result = conv.convert(i32, bool, 1);
    try std.testing.expectEqual(bool, @TypeOf(result));
}

test "convert float to bool" {
    tests_log.info("\n     convert float to bool", .{});
    const result = conv.convert(f64, bool, 0.0);
    try std.testing.expectEqual(false, result);
}

test "convert true bool to integer" {
    tests_log.info("\n     convert bool to integer", .{});
    const result = conv.convert(bool, i32, true);
    try std.testing.expectEqual(i32, @TypeOf(result));
    try std.testing.expectEqual(1, result);
}

test "convert false bool to integer" {
    tests_log.info("\n     convert bool to integer", .{});
    const result = conv.convert(bool, i32, false);
    try std.testing.expectEqual(i32, @TypeOf(result));
    try std.testing.expectEqual(0, result);
}

test "convert bool to float" {
    tests_log.info("\n     convert bool to float", .{});
    const result = conv.convert(bool, f64, false);
    try std.testing.expectEqual(0.0, result);
    try std.testing.expectEqual(f64, @TypeOf(result));
}

test "convert bool to bool" {
    tests_log.info("\n     convert bool to bool", .{});
    const result = conv.convert(bool, bool, true);
    try std.testing.expectEqual(true, result);
    try std.testing.expectEqual(bool, @TypeOf(result));
}

test "convert comptime int to float" {
    tests_log.info("\n     convert comptime int to float", .{});
    comptime {
        const a = 123;
        const result = conv.convert(@TypeOf(a), f64, a);
        try std.testing.expectEqual(123.0, result);
        try std.testing.expectEqual(f64, @TypeOf(result));
    }
}

test "generateRandomSlice allocates correctly" {
    tests_log.info("\n     Checking allocation in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(f32, allocator, 10, init.InitMethod.Dumb);
    defer allocator.free(slice);

    try std.testing.expectEqual(slice.len, 10);
}

test "generateRandomSlice produces only 0 and 1 for Binary" {
    tests_log.info("\n     Checking binary values in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(u8, allocator, 100, init.InitMethod.Binary);
    defer allocator.free(slice);

    for (slice) |val| {
        try std.testing.expect(val == 0 or val == 1);
    }
}

test "generateRandomSlice respects LimitedRange" {
    tests_log.info("\n     Checking limited range in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(i32, allocator, 100, init.InitMethod.LimitedRange);
    defer allocator.free(slice);

    for (slice) |val| {
        try std.testing.expect(val >= 10 and val <= 100);
    }
}

test "generateRandomSlice respects Gaussian distribution" {
    tests_log.info("\n     Checking Gaussian distribution in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(f64, allocator, 10000, init.InitMethod.Gaussian);
    defer allocator.free(slice);

    var sum: f64 = 0;
    for (slice) |val| {
        sum += val;
    }
    const mean = sum / @as(f64, @floatFromInt(slice.len));
    try std.testing.expect(mean > -0.2 and mean < 0.2);
}

// ============ TEST NCHW->NHWC ============

test "from_NCHW_to_NHWC - simple RGB conversion" {
    tests_log.info("\n     convert NCHW to NHWC for RGB image", .{});

    const test_alloc = std.testing.allocator;

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = 1;
    shape[1] = 3;
    shape[2] = 2;
    shape[3] = 2;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const data = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };

    const tensor_f32 = try Tensor(f32).fromArray(&test_alloc, &data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(f32));
    tensor_ptr.* = tensor_f32;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .f32 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_rgb",
        .ty = .f32,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), tensor_nhwc.shape[3]);

    const expected = [_]f32{
        1.0, 5.0, 9.0,
        2.0, 6.0, 10.0,
        3.0, 7.0, 11.0,
        4.0, 8.0, 12.0,
    };

    const result = tensor_nhwc.ptr.?.f32.data;
    for (expected, 0..) |exp, i| {
        try std.testing.expectEqual(exp, result[i]);
    }
}

test "from_NCHW_to_NHWC - batch conversion with i8" {
    tests_log.info("\n     convert NCHW to NHWC for batched i8 data", .{});

    const test_alloc = std.testing.allocator;

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 2;
    shape[3] = 2;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const data = [_]i8{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    const tensor_i8 = try Tensor(i8).fromArray(&test_alloc, &data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(i8));
    tensor_ptr.* = tensor_i8;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .i8 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_batch_i8",
        .ty = .i8,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[3]);

    const expected = [_]i8{
        1,  5,  2,  6,
        3,  7,  4,  8,
        9,  13, 10, 14,
        11, 15, 12, 16,
    };

    const result = tensor_nhwc.ptr.?.i8.data;
    for (expected, 0..) |exp, i| {
        try std.testing.expectEqual(exp, result[i]);
    }
}

test "from_NCHW_to_NHWC - random u8 data" {
    tests_log.info("\n     convert NCHW to NHWC with random u8 values", .{});

    const test_alloc = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    const N: usize = 2;
    const C: usize = 4;
    const H: usize = 3;
    const W: usize = 3;

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = N;
    shape[1] = C;
    shape[2] = H;
    shape[3] = W;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const total_size = N * C * H * W;
    const data = try test_alloc.alloc(u8, total_size);
    defer test_alloc.free(data);

    for (data) |*val| {
        val.* = rand.int(u8);
    }

    const tensor_u8 = try Tensor(u8).fromArray(&test_alloc, data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(u8));
    tensor_ptr.* = tensor_u8;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .u8 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_random_u8",
        .ty = .u8,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(N, tensor_nhwc.shape[0]);
    try std.testing.expectEqual(H, tensor_nhwc.shape[1]);
    try std.testing.expectEqual(W, tensor_nhwc.shape[2]);
    try std.testing.expectEqual(C, tensor_nhwc.shape[3]);

    const result = tensor_nhwc.ptr.?.u8.data;

    for (0..C) |c| {
        const nchw_idx = c * H * W;
        const nhwc_idx = c;
        try std.testing.expectEqual(data[nchw_idx], result[nhwc_idx]);
    }
}

test "from_NCHW_to_NHWC - large batch random f32" {
    tests_log.info("\n     convert NCHW to NHWC with large batch of random f32", .{});

    const test_alloc = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(54321);
    const rand = prng.random();

    const N: usize = 4;
    const C: usize = 16;
    const H: usize = 8;
    const W: usize = 8;

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = N;
    shape[1] = C;
    shape[2] = H;
    shape[3] = W;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const total_size = N * C * H * W;
    const data = try test_alloc.alloc(f32, total_size);
    defer test_alloc.free(data);

    for (data) |*val| {
        val.* = rand.float(f32) * 255.0;
    }

    const tensor_f32 = try Tensor(f32).fromArray(&test_alloc, data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(f32));
    tensor_ptr.* = tensor_f32;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .f32 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_large_batch",
        .ty = .f32,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(N, tensor_nhwc.shape[0]);
    try std.testing.expectEqual(H, tensor_nhwc.shape[1]);
    try std.testing.expectEqual(W, tensor_nhwc.shape[2]);
    try std.testing.expectEqual(C, tensor_nhwc.shape[3]);

    const result = tensor_nhwc.ptr.?.f32.data;

    for (0..N) |n| {
        const center_h = H / 2;
        const center_w = W / 2;

        for (0..C) |c| {
            const nchw_idx = ((n * C + c) * H + center_h) * W + center_w;
            const nhwc_idx = ((n * H + center_h) * W + center_w) * C + c;
            try std.testing.expectEqual(data[nchw_idx], result[nhwc_idx]);
        }
    }
}

test "from_NCHW_to_NHWC - single channel f64" {
    tests_log.info("\n     convert NCHW to NHWC for single channel f64", .{});

    const test_alloc = std.testing.allocator;

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 4;
    shape[3] = 4;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const data = [_]f64{
        1.1,  2.2,  3.3,  4.4,
        5.5,  6.6,  7.7,  8.8,
        9.9,  10.0, 11.1, 12.2,
        13.3, 14.4, 15.5, 16.6,
    };

    const tensor_f64 = try Tensor(f64).fromArray(&test_alloc, &data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(f64));
    tensor_ptr.* = tensor_f64;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .f64 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_single_channel",
        .ty = .f64,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), tensor_nhwc.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), tensor_nhwc.shape[2]);
    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.shape[3]);

    const result = tensor_nhwc.ptr.?.f64.data;
    for (data, 0..) |exp, i| {
        try std.testing.expectEqual(exp, result[i]);
    }
}

test "from_NCHW_to_NHWC - random i32 values" {
    tests_log.info("\n     convert NCHW to NHWC with random i32 values", .{});

    const test_alloc = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(99999);
    const rand = prng.random();

    const shape = try test_alloc.alloc(usize, 4);
    shape[0] = 3;
    shape[1] = 8;
    shape[2] = 4;
    shape[3] = 4;

    const stride = try test_alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const total_size = shape[0] * shape[1] * shape[2] * shape[3];
    const data = try test_alloc.alloc(i32, total_size);
    defer test_alloc.free(data);

    for (data) |*val| {
        val.* = rand.intRangeAtMost(i32, -1000, 1000);
    }

    const tensor_i32 = try Tensor(i32).fromArray(&test_alloc, data, shape);
    const tensor_ptr = try test_alloc.create(Tensor(i32));
    tensor_ptr.* = tensor_i32;

    const any_tensor = try test_alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .i32 = tensor_ptr };

    const tensor_nchw = try test_alloc.create(TensorZant);
    tensor_nchw.* = TensorZant{
        .name = "test_i32",
        .ty = .i32,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    const tensor_nhwc = try from_NCHW_to_NHWC(test_alloc, tensor_nchw);
    defer {
        if (tensor_nhwc.ptr) |any_ptr| {
            switch (any_ptr.*) {
                inline else => |inner_tensor| {
                    inner_tensor.deinit();
                    test_alloc.destroy(inner_tensor);
                },
            }
            test_alloc.destroy(any_ptr);
        }
        test_alloc.free(tensor_nhwc.shape);
        test_alloc.free(tensor_nhwc.stride);
        test_alloc.destroy(tensor_nhwc);
    }

    try std.testing.expectEqual(@as(usize, 3), tensor_nhwc.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), tensor_nhwc.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), tensor_nhwc.shape[2]);
    try std.testing.expectEqual(@as(usize, 8), tensor_nhwc.shape[3]);
}
