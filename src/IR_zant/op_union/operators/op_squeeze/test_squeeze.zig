const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "Squeeze - Null axes (remove all size 1 dimensions)" {
    tests_log.info("\n     test: Squeeze - Null axes (remove all size-1 dimensions) ", .{});
    const allocator = pkgAllocator.allocator;

    // Input shape: [1, 2, 1, 3]
    var inputArray: [1][2][1][3]u8 = [_][2][1][3]u8{[_][1][3]u8{
        [_][3]u8{
            [_]u8{ 1, 2, 3 },
        },
        [_][3]u8{
            [_]u8{ 4, 5, 6 },
        },
    }};
    var shape: [4]usize = [_]usize{ 1, 2, 1, 3 };

    var input = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var output = try TensMath.squeeze(u8, &input, null);
    defer output.deinit();

    // Expected output shape: [2, 3]
    try std.testing.expect(output.shape.len == 2);
    try std.testing.expect(output.shape[0] == 2);
    try std.testing.expect(output.shape[1] == 3);
    try std.testing.expect(output.size == 6);

    // Expected values (unchanged)
    try std.testing.expect(output.data[0] == 1);
    try std.testing.expect(output.data[5] == 6);
}

test "Squeeze - Specific axis" {
    tests_log.info("\n     test: Squeeze - Specific axis ", .{});
    const allocator = pkgAllocator.allocator;

    // Input shape: [1, 2, 1, 3]
    var inputArray: [1][2][1][3]u8 = [_][2][1][3]u8{[_][1][3]u8{
        [_][3]u8{
            [_]u8{ 1, 2, 3 },
        },
        [_][3]u8{
            [_]u8{ 4, 5, 6 },
        },
    }};
    var shape: [4]usize = [_]usize{ 1, 2, 1, 3 };

    var input = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    // Squeeze axis 2
    var output = try TensMath.squeeze(u8, &input, &.{2});
    defer output.deinit();

    // Expected output shape: [1, 2, 3]
    try std.testing.expect(output.shape.len == 3);
    try std.testing.expect(output.shape[0] == 1);
    try std.testing.expect(output.shape[1] == 2);
    try std.testing.expect(output.shape[2] == 3);
    try std.testing.expect(output.size == 6);

    // Expected values (unchanged)
    try std.testing.expect(output.data[0] == 1.0);
    try std.testing.expect(output.data[5] == 6.0);
}

test "Squeeze - Multiple axes" {
    tests_log.info("\n     test: Squeeze - Multiple axes ", .{});
    const allocator = pkgAllocator.allocator;

    // Input shape: [1, 1, 2, 3, 1]
    const inputArray: [1][1][2][3][1]u8 = [_][1][2][3][1]u8{
        [_][2][3][1]u8{[_][3][1]u8{ [_][1]u8{ [_]u8{1}, [_]u8{2}, [_]u8{3} }, [_][1]u8{ [_]u8{4}, [_]u8{5}, [_]u8{6} } }},
    };
    var shape: [5]usize = [_]usize{ 1, 1, 2, 3, 1 };

    var input = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    // Squeeze axes 0, 1, rank-1=4
    var output = try TensMath.squeeze(u8, &input, &.{ 0, 1, -1 });
    defer output.deinit();

    // Expected shape: [2, 3]
    try std.testing.expect(output.shape.len == 2);
    try std.testing.expect(output.shape[0] == 2);
    try std.testing.expect(output.shape[1] == 3);
    try std.testing.expect(output.size == 6);

    // Expected values (unchanged)
    try std.testing.expect(output.data[0] == 1);
    try std.testing.expect(output.data[1] == 2);
    try std.testing.expect(output.data[2] == 3);
    try std.testing.expect(output.data[5] == 6);
}

test "Squeeze - Invalid axis (not size 1)" {
    tests_log.info("\n     test: Squeeze - Invalid axis (not size 1) ", .{});
    const allocator = pkgAllocator.allocator;

    // Input shape: [1, 2, 1, 3]
    var inputArray: [1][2][1][3]u8 = [_][2][1][3]u8{[_][1][3]u8{
        [_][3]u8{
            [_]u8{ 1, 2, 3 },
        },
        [_][3]u8{
            [_]u8{ 4, 5, 6 },
        },
    }};
    var shape: [4]usize = [_]usize{ 1, 2, 1, 3 };

    var input = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    // Try to squeeze axis 1 (but has size 2), should fail
    try std.testing.expectError(TensorMathError.InvalidAxes, TensMath.squeeze(u8, &input, &.{1}));
}

test "Squeeze - Out of bounds axis" {
    tests_log.info("\n     test: Squeeze - Out of bounds axis ", .{});
    const allocator = pkgAllocator.allocator;

    // Input shape: [1, 2]
    const inputArray: [1][2]u8 = [_][2]u8{[_]u8{ 1, 2 }};
    var shape: [2]usize = [_]usize{ 1, 2 };

    var input = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    // Try to squeeze axis 2 (but the rank is 2), should fail
    try std.testing.expectError(TensorMathError.AxisOutOfRange, TensMath.squeeze(u8, &input, &.{2}));

    // Try to squeeze axis rank-3=-1, should fail
    try std.testing.expectError(TensorMathError.AxisOutOfRange, TensMath.squeeze(u8, &input, &.{-3}));
}
