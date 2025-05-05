const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const lowerReduceMean = TensMath.lowerReduceMean;

test "mean" {
    std.debug.print("\n     test:mean", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [1]usize = [_]usize{3}; // 2x3 matrix
    var inputArray: [3]f32 = [_]f32{ 1.0, 2.0, 3.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);

    try std.testing.expect(2.0 == TensMath.mean(f32, &t1));

    t1.deinit();
}

test "reduce_mean" {
    std.debug.print("\n     test:reduce_mean", .{});
    const allocator = pkgAllocator.allocator;

    // Test case 1: 2D tensor, reduce along axis 0
    {
        var shape: [2]usize = [_]usize{ 2, 3 };
        var input_array: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        var axes = [_]i64{0};
        var result = try TensMath.reduce_mean(f32, &t1, &axes, true, false);
        defer result.deinit();

        // Expected: [[2.5, 3.5, 4.5]] (mean along axis 0, keepdims=true)
        try std.testing.expectEqual(@as(usize, 2), result.shape.len);
        try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 3), result.shape[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), result.data[0], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 3.5), result.data[1], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 4.5), result.data[2], 0.0001);
    }

    // Test case 2: 3D tensor, reduce along multiple axes
    {
        var shape: [3]usize = [_]usize{ 2, 2, 2 };
        var input_array: [8]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        var axes = [_]i64{ 1, 2 };
        var result = try TensMath.reduce_mean(f32, &t1, &axes, false, false);
        defer result.deinit();

        // Expected: [2.5, 6.5] (mean along axes 1,2, keepdims=false)
        try std.testing.expectEqual(@as(usize, 1), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), result.data[0], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 6.5), result.data[1], 0.0001);
    }

    // Test case 3: Reduce all dimensions
    {
        var shape: [2]usize = [_]usize{ 2, 2 };
        var input_array: [4]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        var result = try TensMath.reduce_mean(f32, &t1, null, true, false);
        defer result.deinit();

        // Expected: [[2.5]] (mean of all elements, keepdims=true)
        try std.testing.expectEqual(@as(usize, 2), result.shape.len);
        try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), result.data[0], 0.0001);
    }

    // Test case 4: noop_with_empty_axes = true
    {
        var shape: [2]usize = [_]usize{ 2, 2 };
        var input_array: [4]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        var result = try TensMath.reduce_mean(f32, &t1, null, true, true);
        defer result.deinit();

        // Expected: same as input (identity operation)
        try std.testing.expectEqual(t1.shape.len, result.shape.len);
        try std.testing.expectEqual(t1.shape[0], result.shape[0]);
        try std.testing.expectEqual(t1.shape[1], result.shape[1]);
        for (t1.data, 0..) |val, i| {
            try std.testing.expectApproxEqAbs(val, result.data[i], 0.0001);
        }
    }

    // Test case 5: Negative axes
    {
        var shape: [3]usize = [_]usize{ 2, 3, 2 };
        var input_array: [12]f32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        var axes = [_]i64{-1}; // Last dimension
        var result = try TensMath.reduce_mean(f32, &t1, &axes, false, false);
        defer result.deinit();

        // Expected: [[1.5, 3.5, 5.5], [7.5, 9.5, 11.5]]
        try std.testing.expectEqual(@as(usize, 2), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 3), result.shape[1]);

        const expected = [_]f32{ 1.5, 3.5, 5.5, 7.5, 9.5, 11.5 };
        for (expected, 0..) |val, i| {
            try std.testing.expectApproxEqAbs(val, result.data[i], 0.0001);
        }
    }
}

test "reduce_mean_advanced" {
    std.debug.print("\n     test:reduce_mean_advanced", .{});
    const allocator = pkgAllocator.allocator;

    // Test case 1: 3D tensor with non-contiguous axes
    {
        var shape: [3]usize = [_]usize{ 2, 3, 4 };
        var input_array: [24]f32 = [_]f32{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24,
        };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        // Reduce along first and last axes (0 and 2)
        var axes = [_]i64{ 0, 2 };
        var result = try TensMath.reduce_mean(f32, &t1, &axes, true, false);
        defer result.deinit();

        // Expected shape: [1, 3, 1]
        try std.testing.expectEqual(@as(usize, 3), result.shape.len);
        try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 3), result.shape[1]);
        try std.testing.expectEqual(@as(usize, 1), result.shape[2]);

        // Expected values: [[[ 8.5 ], [ 12.5 ], [ 16.5 ]]]
        try std.testing.expectApproxEqAbs(@as(f32, 8.5), result.data[0], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 12.5), result.data[1], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 16.5), result.data[2], 0.0001);
    }

    // Test case 2: 4D tensor with single axis reduction
    {
        var shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
        var input_array: [16]f32 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var t1 = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
        defer t1.deinit();

        // Reduce along axis 1
        var axes = [_]i64{1};
        var result = try TensMath.reduce_mean(f32, &t1, &axes, false, false);
        defer result.deinit();

        // Expected shape: [2, 2, 2]
        try std.testing.expectEqual(@as(usize, 3), result.shape.len);
        try std.testing.expectEqual(@as(usize, 2), result.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), result.shape[1]);
        try std.testing.expectEqual(@as(usize, 2), result.shape[2]);

        // Check first few expected values
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.data[0], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 4.0), result.data[1], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), result.data[2], 0.0001);
    }
}

test "reduce_mean_uops_print" {
    std.debug.print("\n     test:reduce_mean_uops_print", .{});
    const allocator = pkgAllocator.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    defer builder.deinit();

    // 2. Define Input Tensor (as a UOp)
    const in_shape = [_]usize{ 2, 3 };
    const in_stride = [_]isize{ 3, 1 };
    const dtype = DType.f32;

    // Define a dummy input global tensor UOp
    // Duplicate in_shape for the UOp payload to avoid using stack memory
    const persistent_in_shape = builder.alloc.dupe(usize, &in_shape) catch unreachable;
    const X_id = builder.push(.DEFINE_GLOBAL, dtype, &.{}, Any{ .shape = persistent_in_shape });

    // 3. Define Reduction Parameters
    const axes = [_]i64{0}; // Reduce along axis 0
    const keepdims = true;
    const noop_with_empty_axes = false;

    // 4. Call lowerReduceMean
    _ = lowerReduceMean(
        &builder,
        X_id,
        &in_shape,
        &in_stride,
        &axes,
        keepdims,
        noop_with_empty_axes,
        dtype,
    );

    // 5. Print the generated UOps
    std.debug.print("\nGenerated UOps for reduce_mean(shape={any}, axes={any}, keepdims={any}):\n", .{
        in_shape, axes, keepdims,
    });
    const stderr_writer = std.io.getStdErr().writer();
    for (builder.list.items) |uop| {
        try uop.dump(stderr_writer);
    }

    // Manually free memory allocated for UOp args (like .shape slices)
    // because builder.deinit() doesn't handle them.
    for (builder.list.items) |uop| {
        if (uop.arg) |arg| {
            switch (arg) {
                .shape => |shape_slice| allocator.free(shape_slice),
                // Add other cases here if other Any payloads allocate memory
                else => {},
            }
        }
    }

    // Optional: Add assertions here later if needed, for now just print
}
