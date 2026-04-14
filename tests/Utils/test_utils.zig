const std = @import("std");
const IR_zant = @import("IR_zant");
const utils = IR_zant.utils;
const TensorZant = IR_zant.TensorZant;
const TensorCategory = IR_zant.TensorCategory;
const TensorType = IR_zant.tensorZant_lib.TensorType;
const Tensor = IR_zant.core.tensor.Tensor;
const AnyTensor = IR_zant.core.tensor.AnyTensor;

const tests_log = std.log.scoped(.test_utils);

test "Utils description test" {
    tests_log.info("\n--- Running utils test\n", .{});
}

test "convert integer to float" {
    tests_log.info("\n     convert integer to float", .{});
    const result = std.math.lossyCast(f64, @as(i32, 42));
    try std.testing.expectEqual(@as(f64, 42.0), result);
    try std.testing.expectEqual(f64, @TypeOf(result));
}

test "convert float to integer" {
    tests_log.info("\n     convert float to integer", .{});
    const result = std.math.lossyCast(i32, @as(f64, 42.9));
    try std.testing.expectEqual(@as(i32, 42), result);
    try std.testing.expectEqual(i32, @TypeOf(result));
}


test "convert comptime int to float" {
    tests_log.info("\n     convert comptime int to float", .{});
    comptime {
        const result = std.math.lossyCast(f64, 123);
        try std.testing.expectEqual(@as(f64, 123.0), result);
        try std.testing.expectEqual(f64, @TypeOf(result));
    }
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
