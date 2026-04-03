const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensMath = zant.core.tensor.math_standard;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_log);

test "log_standard - basic case" {
    tests_log.info("\n     test: log_standard - basic case", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{4};
    var inputArray = [_]f32{ 1.0, 2.718282, 7.389056, 10.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{ 0.0, 1.0, 2.0, 2.302585 }; // ln(1)=0, ln(e)=1, ln(e²)=2, ln(10)≈2.302585
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{4}, result.shape);
}

test "log_standard - fractional values" {
    tests_log.info("\n     test: log_standard - fractional values", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]f32{ 0.5, 0.1, 0.01 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{ -0.693147, -2.302585, -4.605170 }; // ln(0.5), ln(0.1), ln(0.01)
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, result.shape);
}

test "log_standard - single element" {
    tests_log.info("\n     test: log_standard - single element", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{1};
    var inputArray = [_]f32{2.718282};

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{1.0}; // ln(e) = 1
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{1}, result.shape);
}

test "log_standard - 2D tensor" {
    tests_log.info("\n     test: log_standard - 2D tensor", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{ 2, 3 };
    var inputArray = [_]f32{ 1.0, 2.718282, 7.389056, 10.0, 100.0, 1000.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{ 0.0, 1.0, 2.0, 2.302585, 4.605170, 6.907755 };
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, result.shape);
}

test "log_standard - empty tensor" {
    tests_log.info("\n     test: log_standard - empty tensor", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{0};

    var input = try Tensor(f32).fromShape(&allocator, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    try std.testing.expectEqual(result.data.len, 0);
    try std.testing.expectEqualSlices(usize, &[_]usize{0}, result.shape);
}

test "log_standard - zero input (should handle gracefully)" {
    tests_log.info("\n     test: log_standard - zero input", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]f32{ 1.0, 0.0, 2.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    // ln(0) = -inf in standard math, check how your implementation handles it
    try std.testing.expect(result.data[0] == 0.0); // ln(1) = 0
    try std.testing.expect(std.math.isInf(result.data[1])); // ln(0) = -inf
    try std.testing.expectApproxEqRel(result.data[2], 0.693147, 1e-5); // ln(2)
}

test "log_standard - negative input (should handle gracefully)" {
    tests_log.info("\n     test: log_standard - negative input", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{2};
    var inputArray = [_]f32{ -1.0, 1.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    // ln(negative) = NaN in standard math
    try std.testing.expect(std.math.isNan(result.data[0])); // ln(-1) = NaN
    try std.testing.expect(result.data[1] == 0.0); // ln(1) = 0
}

test "log_standard - f64 support" {
    tests_log.info("\n     test: log_standard - f64 support", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]f64{ 1.0, 2.718281828459045, 10.0 };

    var input = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f64, &input);
    defer result.deinit();

    const expected = [_]f64{ 0.0, 1.0, 2.302585092994046 };
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-10);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, result.shape);
}
