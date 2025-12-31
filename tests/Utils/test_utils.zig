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

test "from_NCHW_to_NHWC - physical data reordering" {
    tests_log.info("\n     from_NCHW_to_NHWC - physical data reordering", .{});

    const alloc = std.testing.allocator;

    // Crea tensor NCHW: [1, 2, 2, 3] = 12 elementi
    // N=1, C=2, H=2, W=3
    const tensor_nchw = try createTensorZant([4]usize{ 1, 2, 2, 3 });

    // Dati originali NCHW (sequenziali: 0,1,2,...)
    // Canale 0: [0,1,2, 3,4,5]    (H=2, W=3)
    // Canale 1: [6,7,8, 9,10,11]  (H=2, W=3)

    const tensor_nhwc = try from_NCHW_to_NHWC(alloc, tensor_nchw);

    defer cleanupConvertedTensor(alloc, tensor_nhwc);

    // Verifica shape NHWC: [1, 2, 3, 2]
    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.shape[0]); // N
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[1]); // H
    try std.testing.expectEqual(@as(usize, 3), tensor_nhwc.shape[2]); // W
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.shape[3]); // C

    // Verifica stride NHWC: [12, 6, 2, 1]
    try std.testing.expectEqual(@as(usize, 12), tensor_nhwc.stride[0]);
    try std.testing.expectEqual(@as(usize, 6), tensor_nhwc.stride[1]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nhwc.stride[2]);
    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.stride[3]);

    // Verifica riorganizzazione FISICA dei dati
    const data_nhwc = switch (tensor_nhwc.ptr.?.*) {
        .f32 => |ptr| ptr.data,
        else => unreachable,
    };

    // Layout NHWC atteso:
    // [n=0, h=0, w=0]: [C0=0,  C1=6]   -> indici 0,1
    // [n=0, h=0, w=1]: [C0=1,  C1=7]   -> indici 2,3
    // [n=0, h=0, w=2]: [C0=2,  C1=8]   -> indici 4,5
    // [n=0, h=1, w=0]: [C0=3,  C1=9]   -> indici 6,7
    // [n=0, h=1, w=1]: [C0=4,  C1=10]  -> indici 8,9
    // [n=0, h=1, w=2]: [C0=5,  C1=11]  -> indici 10,11

    try std.testing.expectEqual(@as(f32, 0.0), data_nhwc[0]); // n=0,h=0,w=0,c=0
    try std.testing.expectEqual(@as(f32, 6.0), data_nhwc[1]); // n=0,h=0,w=0,c=1
    try std.testing.expectEqual(@as(f32, 1.0), data_nhwc[2]); // n=0,h=0,w=1,c=0
    try std.testing.expectEqual(@as(f32, 7.0), data_nhwc[3]); // n=0,h=0,w=1,c=1
    try std.testing.expectEqual(@as(f32, 2.0), data_nhwc[4]); // n=0,h=0,w=2,c=0
    try std.testing.expectEqual(@as(f32, 8.0), data_nhwc[5]); // n=0,h=0,w=2,c=1
    try std.testing.expectEqual(@as(f32, 3.0), data_nhwc[6]); // n=0,h=1,w=0,c=0
    try std.testing.expectEqual(@as(f32, 9.0), data_nhwc[7]); // n=0,h=1,w=0,c=1
    try std.testing.expectEqual(@as(f32, 4.0), data_nhwc[8]); // n=0,h=1,w=1,c=0
    try std.testing.expectEqual(@as(f32, 10.0), data_nhwc[9]); // n=0,h=1,w=1,c=1
    try std.testing.expectEqual(@as(f32, 5.0), data_nhwc[10]); // n=0,h=1,w=2,c=0
    try std.testing.expectEqual(@as(f32, 11.0), data_nhwc[11]); // n=0,h=1,w=2,c=1
}

test "from_NHWC_to_NCHW - physical data reordering" {
    tests_log.info("\n     from_NHWC_to_NCHW - physical data reordering", .{});

    const alloc = std.testing.allocator;

    // Crea un tensore che interpretiamo come NHWC: [1, 2, 2, 3]
    // N=1, H=2, W=2, C=3 (Totale 12 elementi)
    const tensor_nhwc = try createTensorZant([4]usize{ 1, 2, 2, 3 });

    // Esegui la conversione inversa
    const tensor_nchw = try from_NHWC_to_NCHW(alloc, tensor_nhwc);

    defer cleanupConvertedTensor(alloc, tensor_nchw);

    // Verifica lo shape NCHW risultante: [1, 3, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), tensor_nchw.shape[0]); // N
    try std.testing.expectEqual(@as(usize, 3), tensor_nchw.shape[1]); // C
    try std.testing.expectEqual(@as(usize, 2), tensor_nchw.shape[2]); // H
    try std.testing.expectEqual(@as(usize, 2), tensor_nchw.shape[3]); // W

    // Verifica gli stride NCHW: [12, 4, 2, 1]
    try std.testing.expectEqual(@as(usize, 12), tensor_nchw.stride[0]);
    try std.testing.expectEqual(@as(usize, 4), tensor_nchw.stride[1]);
    try std.testing.expectEqual(@as(usize, 2), tensor_nchw.stride[2]);
    try std.testing.expectEqual(@as(usize, 1), tensor_nchw.stride[3]);

    // Verifica la riorganizzazione FISICA dei dati
    const data_nchw = switch (tensor_nchw.ptr.?.*) {
        .f32 => |ptr| ptr.data,
        else => unreachable,
    };

    // Nel layout NHWC originale (sequenziale 0..11):
    // Indice 0: n=0, h=0, w=0, c=0 -> valore 0.0
    // Indice 1: n=0, h=0, w=0, c=1 -> valore 1.0
    // Indice 2: n=0, h=0, w=0, c=2 -> valore 2.0
    // Indice 3: n=0, h=0, w=1, c=0 -> valore 3.0
    // ...

    // Nel layout NCHW risultante ci aspettiamo:
    // Canale 0 (c=0): [0, 3, 6, 9]   (H=2, W=2)
    // Canale 1 (c=1): [1, 4, 7, 10]  (H=2, W=2)
    // Canale 2 (c=2): [2, 5, 8, 11]  (H=2, W=2)

    try std.testing.expectEqual(@as(f32, 0.0), data_nchw[0]); // n=0,c=0,h=0,w=0
    try std.testing.expectEqual(@as(f32, 3.0), data_nchw[1]); // n=0,c=0,h=0,w=1
    try std.testing.expectEqual(@as(f32, 6.0), data_nchw[2]); // n=0,c=0,h=1,w=0
    try std.testing.expectEqual(@as(f32, 9.0), data_nchw[3]); // n=0,c=0,h=1,w=1
    try std.testing.expectEqual(@as(f32, 1.0), data_nchw[4]); // n=0,c=1,h=0,w=0
    try std.testing.expectEqual(@as(f32, 4.0), data_nchw[5]); // n=0,c=1,h=0,w=1
    try std.testing.expectEqual(@as(f32, 7.0), data_nchw[6]); // n=0,c=1,h=1,w=0
    try std.testing.expectEqual(@as(f32, 10.0), data_nchw[7]); // n=0,c=1,h=1,w=1
    try std.testing.expectEqual(@as(f32, 2.0), data_nchw[8]); // n=0,c=2,h=0,w=0
    try std.testing.expectEqual(@as(f32, 5.0), data_nchw[9]); // n=0,c=2,h=0,w=1
    try std.testing.expectEqual(@as(f32, 8.0), data_nchw[10]); // n=0,c=2,h=1,w=0
    try std.testing.expectEqual(@as(f32, 11.0), data_nchw[11]); // n=0,c=2,h=1,w=1
}

test "from_NCHW_to_NHWC - verify contiguous memory layout" {
    tests_log.info("\n     from_NCHW_to_NHWC - verify contiguous memory layout", .{});

    const alloc = std.testing.allocator;
    const N, const C, const H, const W = .{ 1, 3, 2, 2 };

    const tensor_nchw = try createTensorZant([4]usize{ 1, 3, 2, 2 });

    const tensor_nhwc = try from_NCHW_to_NHWC(alloc, tensor_nchw);

    defer cleanupConvertedTensor(alloc, tensor_nhwc);

    const data = switch (tensor_nhwc.ptr.?.*) {
        .f32 => |ptr| ptr.data,
        else => unreachable,
    };

    try std.testing.expectEqual(N * H * W * C, data.len);

    try std.testing.expectEqual(@as(usize, 1), tensor_nhwc.stride[3]);
    try std.testing.expectEqual(C, tensor_nhwc.stride[2]);
    try std.testing.expectEqual(W * C, tensor_nhwc.stride[1]);
    try std.testing.expectEqual(H * W * C, tensor_nhwc.stride[0]);

    // Verifica che i dati siano contigui in memoria
    // Nel layout NHWC, i canali devono essere adiacenti
    for (0..data.len - 1) |i| {
        const ptr_curr = @intFromPtr(&data[i]);
        const ptr_next = @intFromPtr(&data[i + 1]);
        // Ogni elemento f32 è 4 bytes
        try std.testing.expectEqual(ptr_curr + @sizeOf(f32), ptr_next);
    }
}

test "from_NHWC_to_NCHW - verify contiguous memory layout" {
    tests_log.info("\n     from_NHWC_to_NCHW - verify contiguous memory layout", .{});

    const alloc = std.testing.allocator;
    const N, const H, const W, const C = .{ 1, 2, 2, 3 };

    const tensor_nhwc = try createTensorZant([4]usize{ N, H, W, C });

    const tensor_nchw = try from_NHWC_to_NCHW(alloc, tensor_nhwc);

    defer cleanupConvertedTensor(alloc, tensor_nchw);

    const data = switch (tensor_nchw.ptr.?.*) {
        .f32 => |ptr| ptr.data,
        else => unreachable,
    };

    // Verifica dimensione totale (compattezza della memoria)
    try std.testing.expectEqual(N * C * H * W, data.len);

    // Verifica strides per layout NCHW contiguo [N, C, H, W]
    try std.testing.expectEqual(@as(usize, 1), tensor_nchw.stride[3]);
    try std.testing.expectEqual(W, tensor_nchw.stride[2]);
    try std.testing.expectEqual(H * W, tensor_nchw.stride[1]);
    try std.testing.expectEqual(C * H * W, tensor_nchw.stride[0]);

    // Verifica che i dati siano contigui in memoria
    // Nel layout NCHW, la larghezza (W) deve essere adiacente
    for (0..data.len - 1) |i| {
        const ptr_curr = @intFromPtr(&data[i]);
        const ptr_next = @intFromPtr(&data[i + 1]);
        try std.testing.expectEqual(ptr_curr + @sizeOf(f32), ptr_next);
    }
}
