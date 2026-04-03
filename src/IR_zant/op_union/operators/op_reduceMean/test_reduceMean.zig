const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ErrorHandler = error_handler;

const tests_log = std.log.scoped(.test_mean);

// Test per mean_standard e mean_lean
test "mean_standard - basic case" {
    tests_log.info("\n     test: mean_standard - basic case", .{});
    const allocator = std.testing.allocator;
    var shape1 = [_]usize{ 2, 2 };
    var shape2 = [_]usize{ 2, 2 };
    var inputArray1 = [_]f32{ 1, 2, 3, 4 };
    var inputArray2 = [_]f32{ 5, 6, 7, 8 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var inputs = [_]*Tensor(f32){ &t1, &t2 };
    var result = try TensMath.mean_standard(f32, &inputs);
    defer result.deinit();

    const expected = [_]f32{ 3, 4, 5, 6 };
    try std.testing.expectEqualSlices(f32, &expected, result.data);
}

test "mean_lean - basic case" {
    tests_log.info("\n     test: mean_lean - basic case", .{});
    const allocator = pkgAllocator.allocator;
    // Stesso caso base, ma con mean_lean
    var shape1 = [_]usize{ 2, 2 };
    var shape2 = [_]usize{ 2, 2 };
    var shape3 = [_]usize{ 2, 2 };
    var inputArray1 = [_]f32{ 1, 2, 3, 4 };
    var inputArray2 = [_]f32{ 5, 6, 7, 8 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();
    var output = try Tensor(f32).fromShape(&allocator, &shape3);
    defer output.deinit();

    var inputs = [_]*Tensor(f32){ &t1, &t2 };
    try TensMath.mean_lean(f32, &inputs, &output);

    const expected = [_]f32{ 3, 4, 5, 6 };
    try std.testing.expectEqualSlices(f32, &expected, output.data);
}

test "mean_standard - broadcasting" {
    tests_log.info("\n     test: mean_standard - broadcasting", .{});
    const allocator = pkgAllocator.allocator;
    // Test con broadcasting: [1, 3] e [2, 3]
    var shape1 = [_]usize{ 1, 3 };
    var shape2 = [_]usize{ 2, 3 };
    var inputArray1 = [_]f32{ 1, 2, 3 };
    var inputArray2 = [_]f32{ 4, 5, 6, 7, 8, 9 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var inputs = [_]*Tensor(f32){ &t1, &t2 };
    var result = try TensMath.mean_standard(f32, &inputs);
    defer result.deinit();

    const expected = [_]f32{ 2.5, 3.5, 4.5, 4, 5, 6 }; // [1+4, 2+5, 3+6]/2, [1+7, 2+8, 3+9]/2
    try std.testing.expectEqualSlices(f32, &expected, result.data);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, result.shape);
}

test "mean_standard - single input" {
    tests_log.info("\n     test: mean_standard - single input", .{});
    const allocator = pkgAllocator.allocator;
    // Edge case: un solo tensore
    var shape1 = [_]usize{ 2, 2 };
    var inputArray1 = [_]f32{ 1, 2, 3, 4 };
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();

    var inputs = [_]*Tensor(f32){&t1};
    var result = try TensMath.mean_standard(f32, &inputs);
    defer result.deinit();

    const expected = [_]f32{ 1, 2, 3, 4 }; // Media di un solo tensore = tensore stesso
    try std.testing.expectEqualSlices(f32, &expected, result.data);
}

test "mean_standard - empty tensor list" {
    tests_log.info("\n     test: mean_standard - empty tensor list", .{});
    // Edge case: lista vuota
    const inputs = [_]*Tensor(f32){};
    const result = TensMath.mean_standard(f32, &inputs);
    try std.testing.expectError(TensorMathError.EmptyTensorList, result);
}

test "mean_standard - invalid type" {
    tests_log.info("\n     test: mean_standard - invalid type", .{});
    const allocator = pkgAllocator.allocator;
    // Edge case: tipo non supportato (es. i32)
    var shape1 = [_]usize{ 2, 2 };
    var inputArray1 = [_]i32{ 1, 2, 3, 4 };
    var t1 = try Tensor(i32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();

    var inputs = [_]*Tensor(i32){&t1};
    try std.testing.expectError(TensorMathError.InvalidDataType, TensMath.mean_standard(i32, &inputs));

    _ = TensMath.mean_standard(i32, &inputs) catch |err| {
        tests_log.warn("\n     Error: {s}", .{ErrorHandler.errorDetails(err)});
    };
}

//non so se esiste un modo per creare inputs con tensori di tipi diversi senza usare casting,
// col casting mean_standard non rileva l'errore

// test "mean_standard - mismatched types" {
//     const allocator = pkgAllocator.allocator;
//     // Edge case: tipi diversi tra tensori
//     var shape1 = [_]usize{ 2, 2 };
//     var shape2 = [_]usize{ 2, 2 };
//     var inputArray1 = [_]f32{ 1, 2, 3, 4 };
//     var inputArray2 = [_]f64{ 5, 6, 7, 8 };
//     var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
//     var t2 = try Tensor(f64).fromArray(&allocator, &inputArray2, &shape2);

//     var inputs = [_]*Tensor(f32){ &t1, @as(*Tensor(f32), @ptrCast(&t2)) }; // Tipo esplicito f32, ma t2 è f64

//     try std.testing.expectError(TensorMathError.MismatchedDataTypes, TensMath.mean_standard(f32, &inputs));
//     _ = TensMath.mean_standard(f32, &inputs) catch |err| {
//         tests_log.warn("\n     Error: {s}", .{ErrorHandler.errorDetails(err)});
//     };

//     t1.deinit();
//     t2.deinit();
// }

test "mean_standard - multidimensional broadcasting" {
    tests_log.info("\n     test: mean_standard - multidimensional broadcasting", .{});
    const allocator = std.testing.allocator;
    var shape1 = [_]usize{ 1, 1, 2 };
    var shape2 = [_]usize{ 2, 2, 2 };
    var inputArray1 = [_]f32{ 1, 2 };
    var inputArray2 = [_]f32{ 3, 4, 5, 6, 7, 8, 9, 10 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var inputs = [_]*Tensor(f32){ &t1, &t2 };
    var result = try TensMath.mean_standard(f32, &inputs);
    defer result.deinit();

    const expected = [_]f32{ 2, 3, 3, 4, 4, 5, 5, 6 };
    try std.testing.expectEqualSlices(f32, &expected, result.data);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 2 }, result.shape);
}
