const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "pow - mixed precision with edge cases" {
    std.debug.print("\n     test: pow - mixed precision with edge cases", .{});
    const allocator = pkgAllocator.allocator;

    // Base: f32, Exp: i32 (mixed types)
    // Include edge cases: x^0, x^1, x^negative, 0^positive
    var base_array = [_]f32{ 2.0, 3.0, 5.0, 0.0, 10.0 };
    var base_shape = [_]usize{5};

    var exp_array = [_]i32{ 0, 1, 2, 3, -2 };
    var exp_shape = [_]usize{5};

    var baseTensor = try Tensor(f32).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(i32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f32, i32, &baseTensor, &expTensor);
    defer result.deinit();

    // Verify shape
    try std.testing.expectEqualSlices(usize, &[_]usize{5}, result.shape);

    // Verify values
    try std.testing.expectApproxEqAbs(1.0, result.data[0], 1e-5); // 2^0 = 1
    try std.testing.expectApproxEqAbs(3.0, result.data[1], 1e-5); // 3^1 = 3
    try std.testing.expectApproxEqAbs(25.0, result.data[2], 1e-5); // 5^2 = 25
    try std.testing.expectApproxEqAbs(0.0, result.data[3], 1e-5); // 0^3 = 0
    try std.testing.expectApproxEqAbs(0.01, result.data[4], 1e-5); // 10^-2 = 0.01

    // Verify no NaN or Inf
    for (result.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "pow - complex 4D broadcast with mixed precision" {
    std.debug.print("\n     test: pow - complex 4D broadcast with mixed precision", .{});
    const allocator = pkgAllocator.allocator;

    // Base: f64 tensor (3x1x4x1)
    var base_array = [_]f64{
        // Batch 0
        2.0, 3.0, 4.0, 5.0,
        // Batch 1
        2.0, 3.0, 4.0, 5.0,
        // Batch 2
        2.0, 3.0, 4.0, 5.0,
    };
    var base_shape = [_]usize{ 3, 1, 4, 1 };

    // Exp: i32 tensor (1x2x1x3) - testing integer exponents
    var exp_array = [_]i32{
        // Channel 0
        1, 2,  0,
        // Channel 1
        3, -1, -2,
    };
    var exp_shape = [_]usize{ 1, 2, 1, 3 };

    var baseTensor = try Tensor(f64).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(i32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f64, i32, &baseTensor, &expTensor);
    defer result.deinit();

    // Output shape: 3x2x4x3 = 72 elements
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 2, 4, 3 }, result.shape);

    // Batch 0, Channel 0, Spatial 0 (base=2.0, exp=[1,2,0])
    // [2^1, 2^2, 2^0] = [2.0, 4.0, 1.0]
    const idx_000 = 0 * 24 + 0 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(2.0, result.data[idx_000 + 0], 1e-10);
    try std.testing.expectApproxEqAbs(4.0, result.data[idx_000 + 1], 1e-10);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_000 + 2], 1e-10);

    // Batch 0, Channel 1, Spatial 0 (base=2.0, exp=[3,-1,-2])
    // [2^3, 2^-1, 2^-2] = [8.0, 0.5, 0.25]
    const idx_010 = 0 * 24 + 1 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(8.0, result.data[idx_010 + 0], 1e-10);
    try std.testing.expectApproxEqAbs(0.5, result.data[idx_010 + 1], 1e-10);
    try std.testing.expectApproxEqAbs(0.25, result.data[idx_010 + 2], 1e-10);

    // Batch 1, Channel 0, Spatial 3 (base=5.0, exp=[1,2,0])
    // [5^1, 5^2, 5^0] = [5.0, 25.0, 1.0]
    const idx_103 = 1 * 24 + 0 * 12 + 3 * 3;
    try std.testing.expectApproxEqAbs(5.0, result.data[idx_103 + 0], 1e-10);
    try std.testing.expectApproxEqAbs(25.0, result.data[idx_103 + 1], 1e-10);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_103 + 2], 1e-10);

    // Batch 2, Channel 1, Spatial 2 (base=4.0, exp=[3,-1,-2])
    // [4^3, 4^-1, 4^-2] = [64.0, 0.25, 0.0625]
    const idx_212 = 2 * 24 + 1 * 12 + 2 * 3;
    try std.testing.expectApproxEqAbs(64.0, result.data[idx_212 + 0], 1e-10);
    try std.testing.expectApproxEqAbs(0.25, result.data[idx_212 + 1], 1e-10);
    try std.testing.expectApproxEqAbs(0.0625, result.data[idx_212 + 2], 1e-10);

    // Verify no NaN or Inf in entire output
    for (result.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "pow - error case: zero to negative power" {
    std.debug.print("\n     test: pow - error case: zero to negative power", .{});
    const allocator = pkgAllocator.allocator;

    // Base contains zero
    var base_array = [_]f32{ 0.0, 2.0, 0.0 };
    var base_shape = [_]usize{3};

    // Exp contains negative values
    var exp_array = [_]f32{ -1.0, 2.0, -2.0 };
    var exp_shape = [_]usize{3};

    var baseTensor = try Tensor(f32).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(f32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    // This should return an error (0^-1 = division by zero)
    const result = TensMath.pow(f32, f32, &baseTensor, &expTensor);

    // Expect DivisionError
    try std.testing.expectError(TensorMathError.DivisionError, result);
}
