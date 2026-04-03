const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_logical_math);

test "equal() " {
    tests_log.info("\n     test:equal()", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    try std.testing.expect(TensMath.equal(f32, &t1, &t2) == true);

    //wrong data
    var inputArray3: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 333.0 },
    };

    var shape3: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t3 = try Tensor(f32).fromArray(&allocator, &inputArray3, &shape3);
    defer t3.deinit();

    try std.testing.expect(TensMath.equal(f32, &t3, &t2) == false);

    //wrong shape

    var shape4: [2]usize = [_]usize{ 1, 4 }; // 2x2 matrix

    var t4 = try Tensor(f32).fromArray(&allocator, &inputArray3, &shape4);
    defer t4.deinit();

    try std.testing.expect(TensMath.equal(f32, &t4, &t2) == false);
}

test "tests isSafe() method" {
    tests_log.info("\n     test: isSafe() method ", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.isSafe(u8, &tensor);
}

test "tests isSafe() -> TensorError.NotFiniteValue " {
    tests_log.info("\n     test: isSafe()-> TensorError.NotFiniteValue", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 8.0, 6.0 },
    };
    const zero: f64 = 1.0;
    inputArray[1][1] = inputArray[1][1] / (zero - 1.0); //NaN here
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensore = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer tensore.deinit();
    try std.testing.expect(std.math.isNan(inputArray[1][1]) == false);
    try std.testing.expect(std.math.isFinite(inputArray[1][1]) == false);
    try std.testing.expectError(TensorError.NotFiniteValue, TensMath.isSafe(f64, &tensore));
}

test "tests isSafe() -> TensorError.NanValue " {
    tests_log.info("\n     test: isSafe()-> TensorError.NanValue", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, std.math.nan(f64), 6.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensore = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer tensore.deinit();
    try std.testing.expect(std.math.isNan(inputArray[1][1]) == true);
    try std.testing.expectError(TensorError.NanValue, TensMath.isSafe(f64, &tensore));
}
