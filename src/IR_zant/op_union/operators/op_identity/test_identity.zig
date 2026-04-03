const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "identity" {
    tests_log.info("\n test: identity basic operations\n", .{});
    const allocator = pkgAllocator.allocator;
    {
        //check output correctness
        var input_array = [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1, 2, 3 },
                [_]f32{ 4, 5, 6 },
            },
            [_][3]f32{
                [_]f32{ 7, 8, 9 },
                [_]f32{ 10, 11, 12 },
            },
        };
        var shape = [_]usize{ 2, 2, 3 };

        var tensor = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer tensor.deinit();

        var result = try TensMath.identity(f32, &tensor);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 3), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), result.shape[1]);
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]);

        try std.testing.expectEqual(@as(usize, 12), result.size);

        const expected = [_]f32{
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        };
        for (result.data, 0..) |val, i| {
            try std.testing.expectEqual(expected[i], val);
        }

        //check that the output is not the same alias as the input
        try std.testing.expect(result.data.ptr != tensor.data.ptr);

        tests_log.debug("OK: identity test passed.\n", .{});
    }
}

test "get_identity_shape_output returns a copy of the input shape" {
    //3 dimensionional input shape
    const input_shape = [_]usize{ 2, 3, 5 };

    // Call the function under test
    const output_shape = try TensMath.get_identity_output_shape(&input_shape);
    defer pkgAllocator.allocator.free(output_shape);

    // Check the length matches
    try std.testing.expectEqual(@as(usize, input_shape.len), output_shape.len);

    // Check each element was copied correctly
    for (output_shape, 0..) |val, i| {
        try std.testing.expectEqual(input_shape[i], val);
    }
}
