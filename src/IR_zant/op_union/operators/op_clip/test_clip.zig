const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.lib_elementWise);

test "clip f32 basic" {
    tests_log.info("\n     test: clip f32 basic", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{5};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{ -5.0, 0.0, 5.0, 10.0, 15.0 }, &shape);
    defer input.deinit();

    const min_val: f32 = 0.0;
    const max_val: f32 = 10.0;

    // Create scalar tensors as 1-element arrays
    var scalar_shape = [_]usize{1};
    var min_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{min_val}, &scalar_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{max_val}, &scalar_shape);
    defer max_tensor.deinit();

    var output = try TensMath.clip(f32, allocator, &input, &min_tensor, &max_tensor);
    defer output.deinit();

    const expected = [_]f32{ 0.0, 0.0, 5.0, 10.0, 10.0 };
    try std.testing.expectEqualSlices(f32, &expected, output.data);
}

test "clip i32 with defaults" {
    tests_log.info("\n     test: clip i32 with defaults", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{4};
    var input = try Tensor(i32).fromArray(&allocator, &[_]i32{ -100, 0, 100, 200 }, &shape);
    defer input.deinit();

    // Use null for min/max to test defaults (entire i32 range)
    var output = try TensMath.clip(i32, allocator, &input, null, null);
    defer output.deinit();

    const expected = [_]i32{ -100, 0, 100, 200 }; // Should be unchanged
    try std.testing.expectEqualSlices(i32, &expected, output.data);
}

test "clip f32 min only" {
    tests_log.info("\n     test: clip f32 min only", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{4};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{ -5.0, -1.0, 0.0, 5.0 }, &shape);
    defer input.deinit();

    const min_val: f32 = -1.0;
    var scalar_shape = [_]usize{1};
    var min_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{min_val}, &scalar_shape);
    defer min_tensor.deinit();

    var output = try TensMath.clip(f32, allocator, &input, &min_tensor, null);
    defer output.deinit();

    const expected = [_]f32{ -1.0, -1.0, 0.0, 5.0 };
    try std.testing.expectEqualSlices(f32, &expected, output.data);
}

test "clip f32 max only" {
    tests_log.info("\n     test: clip f32 max only", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{4};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{ -5.0, 0.0, 5.0, 15.0 }, &shape);
    defer input.deinit();

    const max_val: f32 = 5.0;
    var scalar_shape = [_]usize{1};
    var max_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{max_val}, &scalar_shape);
    defer max_tensor.deinit();

    var output = try TensMath.clip(f32, allocator, &input, null, &max_tensor);
    defer output.deinit();

    const expected = [_]f32{ -5.0, 0.0, 5.0, 5.0 };
    try std.testing.expectEqualSlices(f32, &expected, output.data);
}

test "clip f32 min > max" {
    tests_log.info("\n     test: clip f32 min > max", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{4};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{ -5.0, 0.0, 5.0, 10.0 }, &shape);
    defer input.deinit();

    const min_val: f32 = 10.0;
    const max_val: f32 = 0.0; // min > max
    var scalar_shape = [_]usize{1};
    var min_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{min_val}, &scalar_shape);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{max_val}, &scalar_shape);
    defer max_tensor.deinit();

    var output = try TensMath.clip(f32, allocator, &input, &min_tensor, &max_tensor);
    defer output.deinit();

    // When min > max, all output values should be min_val (ONNX spec)
    const expected = [_]f32{ 10.0, 10.0, 10.0, 10.0 };
    try std.testing.expectEqualSlices(f32, &expected, output.data);
}

test "clip error: min not scalar" {
    tests_log.info("\n     test: clip error: min not scalar", .{});
    const allocator = pkgAllocator.allocator;
    var shape1 = [_]usize{1};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{1.0}, &shape1);
    defer input.deinit();
    var shape2 = [_]usize{2};
    var min_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{ 0.0, 0.0 }, &shape2);
    defer min_tensor.deinit();

    const result = TensMath.clip(f32, allocator, &input, &min_tensor, null);
    try std.testing.expectError(TensorMathError.InputTensorNotScalar, result);
}

test "clip error: max not scalar" {
    tests_log.info("\n     test: clip error: max not scalar", .{});
    const allocator = pkgAllocator.allocator;
    var shape1 = [_]usize{1};
    var input = try Tensor(f32).fromArray(&allocator, &[_]f32{1.0}, &shape1);
    defer input.deinit();
    var shape2 = [_]usize{2};
    var max_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{ 10.0, 10.0 }, &shape2);
    defer max_tensor.deinit();

    const result = TensMath.clip(f32, allocator, &input, null, &max_tensor);
    try std.testing.expectError(TensorMathError.InputTensorNotScalar, result);
}

test "clip empty input" {
    tests_log.info("\n     test: clip empty input", .{});
    const allocator = pkgAllocator.allocator;
    var input = try Tensor(f32).init(&allocator);
    // No need to deinit empty tensor if init doesn't allocate

    const result = TensMath.clip(f32, allocator, &input, null, null);
    try std.testing.expectError(TensorError.EmptyTensor, result);
}

test "clip with SIMD vector size 1" {
    tests_log.info("\n     test: clip with SIMD vector size 1", .{});
    // Test case for types where SIMD might not be beneficial or available
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{6};
    var input = try Tensor(u8).fromArray(&allocator, &[_]u8{ 10, 50, 100, 150, 200, 250 }, &shape);
    defer input.deinit();

    const min_val: u8 = 50;
    const max_val: u8 = 200;
    var scalar_shape = [_]usize{1};
    var min_tensor = try Tensor(u8).fromArray(&allocator, &[_]u8{min_val}, scalar_shape[0..]);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(u8).fromArray(&allocator, &[_]u8{max_val}, scalar_shape[0..]);
    defer max_tensor.deinit();

    var output = try TensMath.clip(u8, allocator, &input, &min_tensor, &max_tensor);
    defer output.deinit();

    const expected = [_]u8{ 50, 50, 100, 150, 200, 200 };
    try std.testing.expectEqualSlices(u8, &expected, output.data);
}

test "clip f32 large tensor with SIMD" {
    tests_log.info("\n     test: clip f32 large tensor with SIMD", .{});
    const allocator = pkgAllocator.allocator;
    const size = 1024;
    var input_data = try allocator.alloc(f32, size);
    defer allocator.free(input_data);
    var expected_data = try allocator.alloc(f32, size);
    defer allocator.free(expected_data);

    const min_val: f32 = -1.0;
    const max_val: f32 = 1.0;

    for (0..size) |i| {
        const sign: f32 = if (i % 2 == 0) 1.0 else -1.0;
        input_data[i] = sign * @as(f32, @floatFromInt(i)) * 0.1; // Values like 0.0, -0.1, 0.2, -0.3 ...
        const temp = @max(input_data[i], min_val);
        expected_data[i] = @min(temp, max_val);
    }

    var shape = [_]usize{size};
    var input = try Tensor(f32).fromArray(&allocator, input_data, &shape);
    // Don't defer input.deinit as it takes ownership of input_data now
    defer input.deinit(); // Add deferred cleanup to fix memory leak

    var scalar_shape = [_]usize{1};
    var min_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{min_val}, scalar_shape[0..]);
    defer min_tensor.deinit();
    var max_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{max_val}, scalar_shape[0..]);
    defer max_tensor.deinit();

    var output = try TensMath.clip(f32, allocator, &input, &min_tensor, &max_tensor);
    defer output.deinit(); // This will free input_data

    try std.testing.expectEqualSlices(f32, expected_data, output.data);
}
