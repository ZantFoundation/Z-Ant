const std = @import("std");
const zant = @import("zant");

const QuantMath = zant.core.tensor.quantized_math;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator.allocator;

const testing = std.testing;

// helper

/// Verifies that the scale factor and zero point are correctly computed for the quantized tensor.
/// Params:
/// - tensor: quantized tensor to be tested
/// - max_float_value: max value in the floating point range
/// - min_float_value: min value in the floating point range
pub fn verify_quantization_params(comptime T: type, comptime U: type, tensor: *Tensor(U), max_float_value: T, min_float_value: T) !void {

    // Scale factor testing
    const scale = try tensor.get_scale_factor();
    const integer_range = std.math.maxInt(U) - std.math.minInt(U);
    const quantized_Range = @as(T, @floatFromInt(@as(i32, @intCast(integer_range))));
    const expected_scale = (max_float_value - min_float_value) / quantized_Range;

    // Zero point testing
    const expected_zero = @as(i32, @intCast(@as(i32, std.math.minInt(U)) - @as(i32, @intFromFloat(min_float_value / scale))));

    try testing.expectEqual(expected_scale, scale);
    try testing.expectEqual(expected_zero, try tensor.get_zero_point());
}

// asymmetric quantization tests

test "asymm signed quantization" {
    std.debug.print("\n    test: asymm signed quantization\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct quantization parameters
    const max_float_value = 0.8;
    const min_float_value = -0.5;
    try verify_quantization_params(f32, i8, &outputTensor, max_float_value, min_float_value);
}

test "asymm unsigned quantization" {
    std.debug.print("\n    test: asymm unsigned quantization\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, u8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct quantization parameters
    const max_float_value = 0.8;
    const min_float_value = -0.5;
    try verify_quantization_params(f32, u8, &outputTensor, max_float_value, min_float_value);
}

test "Quantization with uniform input, scale factor equals 0" {
    std.debug.print("\n    test: quantization uniform input, scale equals 0\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 12.0, 12.0, 12.0 },
        [_]f32{ 12.0, 12.0, 12.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // Scale factor and zero point testing
    const expected_zero = @as(i32, @intCast(@as(i32, std.math.minInt(i8)) - @as(i32, @intFromFloat(12.0))));

    try testing.expectEqual(0, try outputTensor.get_scale_factor());
    try testing.expectEqual(expected_zero, try outputTensor.get_zero_point());
}

test "Quantization with uniform input except from one value" {
    std.debug.print("\n    test: Quantization with uniform input except from one value\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 12.0, 12.0, 0.0 },
        [_]f32{ 12.0, 12.0, 12.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct zero dequantization
    const scale = try outputTensor.get_scale_factor();
    const zero = @as(f32, @floatFromInt(try outputTensor.get_zero_point()));
    const quantZero = @as(f32, @floatFromInt(@as(i32, @intCast(try outputTensor.get(2)))));

    const dequantZero = scale * (quantZero - zero);

    try testing.expectEqual(0, dequantZero);

    // test correct quantization parameters
    const max_float_value = 12.0;
    const min_float_value = 0.0;
    try verify_quantization_params(f32, i8, &outputTensor, max_float_value, min_float_value);
}

test "zero dequantization wide range value tensor" {
    std.debug.print("\n    test: test zero dequantization, wide range value tensor\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 120.0, 5e3, 0.0 },
        [_]f32{ -1.0, -1e-5, 170.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct zero dequantization
    const scale = try outputTensor.get_scale_factor();
    const zero = @as(f32, @floatFromInt(try outputTensor.get_zero_point()));
    // get the value corresponding to the 0.0 value, index is 2
    const quantZero = @as(f32, @floatFromInt(@as(i32, @intCast(try outputTensor.get(2)))));

    const dequantZero = scale * (quantZero - zero);

    try testing.expectEqual(0, dequantZero);

    // test correct quantization parameters
    const max_float_value = 5e3;
    const min_float_value = -1.0;
    try verify_quantization_params(f32, i8, &outputTensor, max_float_value, min_float_value);
}

test "only 0 value quantization" {
    std.debug.print("\n    test: quantization only 0 value\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 0.0, 0.0 },
        [_]f32{ 0.0, 0.0, 0.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct zero dequantization
    const scale = try outputTensor.get_scale_factor();
    const zero = @as(f32, @floatFromInt(try outputTensor.get_zero_point()));
    const quantZero = @as(f32, @floatFromInt(@as(i32, @intCast(try outputTensor.get(2)))));

    const dequantZero = scale * (quantZero - zero);

    try testing.expectEqual(0, dequantZero);

    // test correct scale factor
    const integer_range = std.math.maxInt(i8) - std.math.minInt(i8);
    const quantized_Range = @as(f32, @floatFromInt(@as(i32, @intCast(integer_range))));
    const max_float_value = 0.0;
    const min_float_value = 0.0;
    const expected_scale = (max_float_value - min_float_value) / quantized_Range;

    // test correct zero point
    const expected_zero = @as(i32, @intCast(@as(i32, std.math.minInt(i8)) - @as(i32, @intFromFloat(0.0))));

    try testing.expectEqual(expected_scale, scale);
    try testing.expectEqual(expected_zero, try outputTensor.get_zero_point());
}

test "asymm quantization of around zero balanced tensor" {
    std.debug.print("\n    test: asymm quantization of around zero balanced tensor\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -100.0, 0.0, -100.0 },
        [_]f32{ 50.0, -50.0, 100.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct zero dequantization
    const scale = try outputTensor.get_scale_factor();
    const zero = @as(f32, @floatFromInt(try outputTensor.get_zero_point()));
    const quantZero = @as(f32, @floatFromInt(@as(i32, @intCast(try outputTensor.get(1)))));

    const dequantZero = scale * (quantZero - zero);
    try testing.expectEqual(0, dequantZero);

    // test correct scale factor
    const max_float_value = 100.0;
    const min_float_value = -100.0;
    try verify_quantization_params(f32, i8, &outputTensor, max_float_value, min_float_value);
}

test "asymm quantization values at the edge of the input type range" {
    std.debug.print("\n    test: asymm quantization values at the edge of the input type range\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ std.math.floatMax(f32), 0.0, std.math.floatMax(f32) },
        [_]f32{ std.math.floatMin(f32), std.math.floatMin(f32), std.math.floatMin(f32) },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    // test correct zero dequantization
    const scale = try outputTensor.get_scale_factor();
    const zero = @as(f32, @floatFromInt(try outputTensor.get_zero_point()));
    const quantZero = @as(f32, @floatFromInt(@as(i32, @intCast(try outputTensor.get(1)))));

    const dequantZero = scale * (quantZero - zero);
    try testing.expectEqual(0, dequantZero);

    // quantization parameters testing
    const max_float_value = std.math.floatMax(f32);
    const min_float_value = std.math.floatMin(f32);
    try verify_quantization_params(f32, i8, &outputTensor, max_float_value, min_float_value);
}

test "asymm quantization tensor with small scale factor" {
    std.debug.print("\n    test: asymm quantization tensor with small scale factor\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.000000, 1.000000, 1.000000 },
        [_]f32{ 1.000001, 1.000001, 1.000001 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    try verify_quantization_params(f32, i8, &outputTensor, 1.000001, 1.0);
}
