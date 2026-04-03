const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_lib_shape);

test "get_resize_output_shape()" {
    tests_log.info("\n     test: get_resize_output_shape \n", .{});

    var input_shape = [_]usize{ 2, 3, 4 };
    var scales = [_]f32{ 2.0, 1.5, 0.5 };
    var target_sizes = [_]usize{ 4, 4, 2 };

    // Test with scales
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, &scales, null);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test with target sizes
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, null, &target_sizes);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test invalid input (both scales and sizes null)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, null));

    // Test invalid input (both scales and sizes provided)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &scales, &target_sizes));

    // Test mismatched dimensions
    var wrong_scales = [_]f32{ 2.0, 1.5 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &wrong_scales, null));

    var wrong_sizes = [_]usize{ 4, 4 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, &wrong_sizes));
}

test "resize with nearest neighbor interpolation" {
    tests_log.info("\n     test: resize with nearest neighbor interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "nearest", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[0]);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[1]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[2]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[3]);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "nearest", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with linear interpolation" {
    tests_log.info("\n     test: resize with linear interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "linear", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "linear", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with cubic interpolation" {
    tests_log.info("\n     test: resize with cubic interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var shape_1d = [_]usize{8};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale down by 0.5x
    var scales = [_]f32{0.5};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "cubic", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 4), resized_1d.size);
}

test "resize with explicit sizes" {
    tests_log.info("\n     test: resize with explicit sizes", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Resize to specific dimensions
    var sizes = [_]usize{ 3, 3 };
    var resized = try TensMath.resize(u8, &tensor, "nearest", null, &sizes, "half_pixel");
    defer resized.deinit();

    try std.testing.expectEqual(@as(usize, 9), resized.size);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[1]);
}

test "resize error cases" {
    tests_log.info("\n     test: resize error cases", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test invalid mode
    var scales = [_]f32{ 2.0, 2.0 };
    try std.testing.expectError(
        TensorError.UnsupportedMode,
        TensMath.resize(u8, &tensor, "invalid_mode", &scales, null, "half_pixel"),
    );

    // Test both scales and sizes provided
    var sizes = [_]usize{ 3, 3 };
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", &scales, &sizes, "half_pixel"),
    );

    // Test neither scales nor sizes provided
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", null, null, "half_pixel"),
    );
}
