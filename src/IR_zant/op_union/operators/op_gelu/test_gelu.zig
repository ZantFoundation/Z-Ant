const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.lib_elementWise);

test "test gelu with approximate = none and valid f32 tensor" {
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 1.0, -1.0 },
        [_]f32{ 0.5, -0.5, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var result = try TensMath.gelu(f32, &tensor, "none");
    defer result.deinit();

    const expected_values = [_]f32{
        0.0000000000000000,
        0.8413447460685429,
        -0.15865525393145707,
        0.34573123063700656,
        -0.15426876936299344,
        1.9544997361036416,
    };

    const epsilon: f32 = 1e-6;
    for (0..result.size) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, result.data[i], expected_values[i], epsilon));
    }
}

test "test gelu with approximate = tanh and valid f32 tensor" {
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 1.0, -1.0 },
        [_]f32{ 0.5, -0.5, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var result = try TensMath.gelu(f32, &tensor, "tanh");
    defer result.deinit();

    const expected_values = [_]f32{
        0.00000000, // GELU(0.0) = 0.5 * 0 * (1 + tanh(...)) = 0
        0.8411919906082768, // GELU(1.0) ≈ 0.84119225
        -0.15880800939172324, // GELU(-1.0) ≈ -0.15880775
        0.34571400982514394, // GELU(0.5) ≈ 0.34567261
        -0.15428599017485606, // GELU(-0.5) ≈ -0.15432739
        1.954597694087775, // GELU(2.0) ≈ 1.95450306
    };

    const epsilon: f32 = 1e-5;
    for (0..result.size) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, result.data[i], expected_values[i], epsilon));
    }
}
