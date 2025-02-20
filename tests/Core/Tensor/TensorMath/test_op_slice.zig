const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;
const TensorMathError = @import("errorHandler").TensorMathError;
const PoolingType = @import("layer").poolingLayer.PoolingType;

test "slice basic slicing" {
    std.debug.print("\n     test: slice basic slicing", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor slicing
    var input_array_1d = [_]i32{ 1, 2, 3, 4, 5 };
    var shape_1d = [_]usize{5};
    var tensor_1d = try Tensor(i32).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Basic slice [1:3]
    const starts = [_]i64{1};
    const ends = [_]i64{3};
    var sliced_1d = try TensMath.slice(i32, &tensor_1d, &starts, &ends, null, null);
    defer sliced_1d.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced_1d.size);
    try std.testing.expectEqual(@as(i32, 2), sliced_1d.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced_1d.data[1]);

    // Test 2D tensor slicing
    var input_array_2d = [_][3]i32{
        [_]i32{ 1, 2, 3 },
        [_]i32{ 4, 5, 6 },
        [_]i32{ 7, 8, 9 },
    };
    var shape_2d = [_]usize{ 3, 3 };
    var tensor_2d = try Tensor(i32).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Slice [0:2, 1:3]
    const starts_2d = [_]i64{ 0, 1 };
    const ends_2d = [_]i64{ 2, 3 };
    var sliced_2d = try TensMath.slice(i32, &tensor_2d, &starts_2d, &ends_2d, null, null);
    defer sliced_2d.deinit();

    try std.testing.expectEqual(@as(usize, 4), sliced_2d.size);
    try std.testing.expectEqual(@as(i32, 2), sliced_2d.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced_2d.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced_2d.data[2]);
    try std.testing.expectEqual(@as(i32, 6), sliced_2d.data[3]);
}

test "slice negative indices" {
    std.debug.print("\n     test: slice negative indices", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5 };
    var shape = [_]usize{5};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test negative indices [-3:-1]
    const starts = [_]i64{-3};
    const ends = [_]i64{-1};
    var sliced = try TensMath.slice(i32, &tensor, &starts, &ends, null, null);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced.size);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 4), sliced.data[1]);
}

test "slice with steps" {
    std.debug.print("\n     test: slice with steps", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{6};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test with step 2
    var starts = [_]i64{0};
    var ends = [_]i64{6};
    var steps = [_]i64{2};
    var sliced = try TensMath.slice(i32, &tensor, &starts, &ends, null, &steps);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 3), sliced.size);
    try std.testing.expectEqual(@as(i32, 1), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced.data[2]);

    // Test with negative step
    steps[0] = -1;
    starts[0] = 5;
    ends[0] = -1;
    var reversed = try TensMath.slice(i32, &tensor, &starts, &ends, null, &steps);
    defer reversed.deinit();

    try std.testing.expectEqual(@as(usize, 5), reversed.size);
    try std.testing.expectEqual(@as(i32, 6), reversed.data[0]);
    try std.testing.expectEqual(@as(i32, 5), reversed.data[1]);
    try std.testing.expectEqual(@as(i32, 4), reversed.data[2]);
    try std.testing.expectEqual(@as(i32, 3), reversed.data[3]);
    try std.testing.expectEqual(@as(i32, 2), reversed.data[4]);
}

test "slice with explicit axes" {
    std.debug.print("\n     test: slice with explicit axes", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][3]i32{
        [_]i32{ 1, 2, 3 },
        [_]i32{ 4, 5, 6 },
        [_]i32{ 7, 8, 9 },
    };
    var shape = [_]usize{ 3, 3 };
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test slicing only along axis 1
    const starts = [_]i64{1};
    const ends = [_]i64{3};
    const axes = [_]i64{1};
    var sliced = try TensMath.slice(i32, &tensor, &starts, &ends, &axes, null);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), sliced.shape[0]);
    try std.testing.expectEqual(@as(i32, 2), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced.data[2]);
    try std.testing.expectEqual(@as(i32, 6), sliced.data[3]);
    try std.testing.expectEqual(@as(i32, 8), sliced.data[4]);
    try std.testing.expectEqual(@as(i32, 9), sliced.data[5]);
}

test "slice error cases" {
    std.debug.print("\n     test: slice error cases", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5 };
    var shape = [_]usize{5};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test invalid step size
    const starts = [_]i64{0};
    const ends = [_]i64{5};
    const steps = [_]i64{0}; // Step cannot be 0
    try std.testing.expectError(TensorMathError.InvalidSliceStep, TensMath.slice(i32, &tensor, &starts, &ends, null, &steps));

    // Test mismatched lengths
    const starts_2 = [_]i64{ 0, 0 };
    const ends_1 = [_]i64{5};
    try std.testing.expectError(TensorMathError.InvalidSliceIndices, TensMath.slice(i32, &tensor, &starts_2, &ends_1, null, null));

    // Test invalid axis
    const axes = [_]i64{5}; // Axis 5 doesn't exist in a 1D tensor
    try std.testing.expectError(TensorMathError.InvalidSliceIndices, TensMath.slice(i32, &tensor, &starts, &ends, &axes, null));
}
