const std = @import("std");
const zant = @import("zant");
const conv = zant.utils.type_converter;
const init = zant.utils.tensor_initializer;

// Aggiungi questi import per i test NCHW->NHWC:
const IR_zant = @import("IR_zant");
const utils = IR_zant.utils;
const TensorZant = IR_zant.TensorZant;
const TensorCategory = IR_zant.TensorCategory;
const TensorType = IR_zant.tensorZant_lib.TensorType;
const Tensor = zant.core.tensor.Tensor;
const AnyTensor = zant.core.tensor.AnyTensor;

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

// ============ HELPER FUNCTION PER CREARE TENSORZANT ============

/// Helper semplificato: crea un TensorZant f32 con dati sequenziali
fn createTensorZant(shape_vals: [4]usize) !*TensorZant {
    const alloc = std.testing.allocator;

    const shape = try alloc.alloc(usize, 4);
    @memcpy(shape, &shape_vals);

    // Calcola dimensione totale
    const total_size = shape_vals[0] * shape_vals[1] * shape_vals[2] * shape_vals[3];

    // Crea dati sequenziali: [0.0, 1.0, 2.0, ..., n-1.0]
    const data = try alloc.alloc(f32, total_size);
    for (data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    defer alloc.free(data);

    const stride = try alloc.alloc(usize, 4);
    stride[3] = 1;
    stride[2] = shape[3];
    stride[1] = shape[2] * stride[2];
    stride[0] = shape[1] * stride[1];

    const tensor = try Tensor(f32).fromArray(&alloc, data, shape);
    const tensor_ptr = try alloc.create(Tensor(f32));
    tensor_ptr.* = tensor;

    const any_tensor = try alloc.create(AnyTensor);
    any_tensor.* = AnyTensor{ .f32 = tensor_ptr };

    const tensor_zant = try alloc.create(TensorZant);
    tensor_zant.* = TensorZant{
        .name = "test_tensor",
        .ty = TensorType.f32,
        .tc = TensorCategory.LINK,
        .ptr = any_tensor,
        .shape = shape,
        .stride = stride,
    };

    return tensor_zant;
}

fn cleanupTensorZant(alloc: std.mem.Allocator, tensor: *TensorZant) void {
    if (tensor.ptr) |any_ptr| {
        switch (any_ptr.*) {
            inline else => |inner_tensor| {
                inner_tensor.deinit();
                alloc.destroy(inner_tensor);
            },
        }
        alloc.destroy(any_ptr);
    }
    alloc.free(tensor.shape);
    alloc.free(tensor.stride);
    alloc.destroy(tensor);
}

fn cleanupConvertedTensor(alloc: std.mem.Allocator, tensor: *TensorZant) void {
    if (tensor.ptr) |any_ptr| {
        switch (any_ptr.*) {
            inline else => |inner_tensor| {
                // Dealloca SOLO i dati, NON lo shape
                const tensor_alloc = inner_tensor.allocator.*;
                tensor_alloc.free(inner_tensor.data);
                // ❌ NON chiamare inner_tensor.deinit() - dealloca anche lo shape
                // ❌ NON deallocare inner_tensor.shape - è lo stesso di tensor.shape
                alloc.destroy(inner_tensor);
            },
        }
        alloc.destroy(any_ptr);
    }
    // ✅ Dealloca shape (questo è lo shape allocato in convertNCHWtoNHWC)
    alloc.free(tensor.shape);
    alloc.free(tensor.stride);
    alloc.destroy(tensor);
}
