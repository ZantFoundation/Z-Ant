const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

test "get_pads_output_shape - basic" {
    tests_log.info("\n     test: get_pads_output_shape - basic", .{});
    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 3, 4 };
    var pads = [_]i64{ 1, 1, 2, 2 }; // [x1_begin, x2_begin, x1_end, x2_end]

    const output_shape = try TensMath.get_pads_output_shape(allocator, &input_shape, &pads, null);
    defer allocator.free(output_shape);

    try std.testing.expectEqual(@as(usize, 2), output_shape.len);
    try std.testing.expectEqual(@as(usize, 3 + 1 + 2), output_shape[0]); // 3 + pad_start[0] + pad_end[0]
    try std.testing.expectEqual(@as(usize, 4 + 1 + 2), output_shape[1]); // 4 + pad_start[1] + pad_end[1]
}

test "get_pads_output_shape - with axes" {
    tests_log.info("\n     test: get_pads_output_shape - with axes", .{});
    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 3, 4, 5 };
    var pads = [_]i64{ 1, 2, 3, 4 }; // pads for axes 0 and 2: [p0_start, p2_start, p0_end, p2_end]
    var axes = [_]isize{ 0, 2 };

    const output_shape = try TensMath.get_pads_output_shape(allocator, &input_shape, &pads, &axes);
    defer allocator.free(output_shape);

    try std.testing.expectEqual(@as(usize, 3), output_shape.len);
    try std.testing.expectEqual(@as(usize, 3 + 1 + 3), output_shape[0]); // Axis 0 padded
    try std.testing.expectEqual(@as(usize, 4), output_shape[1]); // Axis 1 not padded
    try std.testing.expectEqual(@as(usize, 5 + 2 + 4), output_shape[2]); // Axis 2 padded
}

test "get_pads_output_shape - negative axes" {
    tests_log.info("\n     test: get_pads_output_shape - negative axes", .{});
    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 3, 4, 5 };
    var pads = [_]i64{ 1, 2 }; // pads for axis -1 (axis 2): [p2_start, p2_end]
    var axes = [_]isize{-1};

    const output_shape = try TensMath.get_pads_output_shape(allocator, &input_shape, &pads, &axes);
    defer allocator.free(output_shape);

    try std.testing.expectEqual(@as(usize, 3), output_shape.len);
    try std.testing.expectEqual(@as(usize, 3), output_shape[0]); // Axis 0 not padded
    try std.testing.expectEqual(@as(usize, 4), output_shape[1]); // Axis 1 not padded
    try std.testing.expectEqual(@as(usize, 5 + 1 + 2), output_shape[2]); // Axis 2 padded
}

test "get_pads_output_shape - error cases" {
    tests_log.info("\n     test: get_pads_output_shape - error cases", .{});
    const allocator = pkgAllocator.allocator;
    var input_shape = [_]usize{ 3, 4 };

    // Invalid pads length (should be rank * 2)
    var invalid_pads = [_]i64{ 1, 1, 2 };
    try std.testing.expectError(TensorMathError.InvalidPaddingShape, TensMath.get_pads_output_shape(allocator, &input_shape, &invalid_pads, null));

    // Invalid axes length (pads length must be axes.len * 2)
    var pads = [_]i64{ 1, 2 };
    var invalid_axes = [_]isize{ 0, 1 }; // axes.len = 2, but pads.len = 2 != 2*2
    try std.testing.expectError(TensorMathError.InvalidPaddingShape, TensMath.get_pads_output_shape(allocator, &input_shape, &pads, &invalid_axes));

    // Repeated axes
    var pads_rep = [_]i64{ 1, 1, 2, 2 };
    var repeated_axes = [_]isize{ 0, 0 };
    try std.testing.expectError(TensorMathError.InvalidInput, TensMath.get_pads_output_shape(allocator, &input_shape, &pads_rep, &repeated_axes));

    // Axis out of range
    var pads_oor = [_]i64{ 1, 2 };
    var oor_axes = [_]isize{2}; // axis 2 is out of range for rank 2
    try std.testing.expectError(TensorMathError.AxisOutOfRange, TensMath.get_pads_output_shape(allocator, &input_shape, &pads_oor, &oor_axes));

    // Padding results in non-positive dimension
    var neg_pads = [_]i64{ -2, -2, -2, -2 };
    try std.testing.expectError(TensorMathError.InvalidPaddingSize, TensMath.get_pads_output_shape(allocator, &input_shape, &neg_pads, null));
}

// --- Pad Tests --- //

// Helper to compare tensors for pad tests
fn expectTensorEqual(comptime T: type, expected: *const Tensor(T), actual: *const Tensor(T)) !void {
    try std.testing.expectEqual(expected.shape.len, actual.shape.len);
    for (expected.shape, 0..) |dim, i| {
        try std.testing.expectEqual(dim, actual.shape[i]);
    }
    try std.testing.expectEqual(expected.size, actual.size);
    for (expected.data, 0..) |val, i| {
        if (@TypeOf(T) == f32 or @TypeOf(T) == f64) {
            try std.testing.expectApproxEqAbs(val, actual.data[i], 1e-6);
        } else {
            try std.testing.expectEqual(val, actual.data[i]);
        }
    }
}

// Test Case 1: Constant Mode (like ONNX Example 1)
test "pads constant mode basic 2D" {
    tests_log.info("\n     test: pads constant mode basic 2D", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_][2]f32{
        [_]f32{ 1.0, 1.2 },
        [_]f32{ 2.3, 3.4 },
        [_]f32{ 4.5, 5.7 },
    };
    var data_shape = [_]usize{ 3, 2 };
    var data_tensor = try Tensor(f32).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    var pads_array = [_]i64{ 0, 2, 0, 0 }; // pads = [x0_begin, x1_begin, x0_end, x1_end]
    var pads_shape = [_]usize{4};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var output = try TensMath.pads(f32, &data_tensor, &pads_tensor, "constant", @as(f32, 0.0), null);
    defer output.deinit();

    var expected_array = [_][4]f32{
        [_]f32{ 0.0, 0.0, 1.0, 1.2 },
        [_]f32{ 0.0, 0.0, 2.3, 3.4 },
        [_]f32{ 0.0, 0.0, 4.5, 5.7 },
    };
    var expected_shape = [_]usize{ 3, 4 };
    var expected_tensor = try Tensor(f32).fromArray(&allocator, &expected_array, &expected_shape);
    defer expected_tensor.deinit();

    try expectTensorEqual(f32, &expected_tensor, &output);
}

// Test Case 2: Constant Mode with specific value and axes
test "pads constant mode with value and axes" {
    tests_log.info("\n     test: pads constant mode with value and axes", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_]i32{ 1, 2, 3 };
    var data_shape = [_]usize{3};
    var data_tensor = try Tensor(i32).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    var pads_array = [_]i64{ 2, 1 }; // pads for axis 0: [p0_start, p0_end]
    var pads_shape = [_]usize{2};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var axes_array = [_]isize{0};
    var axes_shape = [_]usize{1};
    var axes_tensor = try Tensor(isize).fromArray(&allocator, &axes_array, &axes_shape);
    defer axes_tensor.deinit();

    var output = try TensMath.pads(i32, &data_tensor, &pads_tensor, "constant", @as(i32, -5), &axes_tensor);
    defer output.deinit();

    var expected_array = [_]i32{ -5, -5, 1, 2, 3, -5 };
    var expected_shape = [_]usize{6};
    var expected_tensor = try Tensor(i32).fromArray(&allocator, &expected_array, &expected_shape);
    defer expected_tensor.deinit();

    try expectTensorEqual(i32, &expected_tensor, &output);
}

// Test Case 4: Edge Mode (like ONNX Example 3)
test "pads edge mode basic 2D" {
    tests_log.info("\n     test: pads edge mode basic 2D", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_][2]f32{
        [_]f32{ 1.0, 1.2 },
        [_]f32{ 2.3, 3.4 },
        [_]f32{ 4.5, 5.7 },
    };
    var data_shape = [_]usize{ 3, 2 };
    var data_tensor = try Tensor(f32).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    var pads_array = [_]i64{ 0, 2, 0, 0 }; // pads = [x0_begin, x1_begin, x0_end, x1_end]
    var pads_shape = [_]usize{4};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var output = try TensMath.pads(f32, &data_tensor, &pads_tensor, "edge", null, null);
    defer output.deinit();

    var expected_array = [_][4]f32{
        [_]f32{ 1.0, 1.0, 1.0, 1.2 }, // Prepend edge value 1.0 twice
        [_]f32{ 2.3, 2.3, 2.3, 3.4 }, // Prepend edge value 2.3 twice
        [_]f32{ 4.5, 4.5, 4.5, 5.7 }, // Prepend edge value 4.5 twice
    };
    var expected_shape = [_]usize{ 3, 4 };
    var expected_tensor = try Tensor(f32).fromArray(&allocator, &expected_array, &expected_shape);
    defer expected_tensor.deinit();

    try expectTensorEqual(f32, &expected_tensor, &output);
}

// Test Case 5: Wrap Mode (like ONNX Example 4)
test "pads wrap mode basic 2D" {
    tests_log.info("\n     test: pads wrap mode basic 2D", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_][2]f32{
        [_]f32{ 1.0, 1.2 }, // Row 0
        [_]f32{ 2.3, 3.4 }, // Row 1
        [_]f32{ 4.5, 5.7 }, // Row 2
    };
    var data_shape = [_]usize{ 3, 2 };
    var data_tensor = try Tensor(f32).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    // Pads: [pad_start_0, pad_start_1, pad_end_0, pad_end_1]
    var pads_array = [_]i64{ 1, 1, 1, 2 }; // Pad row 0: 1 start, 1 end; Pad row 1: 1 start, 2 end
    var pads_shape = [_]usize{4};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var output = try TensMath.pads(f32, &data_tensor, &pads_tensor, "wrap", null, null);
    defer output.deinit();

    // Expected Output Shape: [3 + 1 + 1, 2 + 1 + 2] = [5, 5]
    // Row padding wraps rows: prepend row 2, append row 0
    // Col padding wraps cols: prepend col 1, append cols 0, 1

    // Expected Data (5x5):
    // Row -1 (wrap row 2): [4.5, 5.7] -> Col padded: [5.7, 4.5, 5.7, 4.5, 5.7]
    // Row 0 (data row 0):  [1.0, 1.2] -> Col padded: [1.2, 1.0, 1.2, 1.0, 1.2]
    // Row 1 (data row 1):  [2.3, 3.4] -> Col padded: [3.4, 2.3, 3.4, 2.3, 3.4]
    // Row 2 (data row 2):  [4.5, 5.7] -> Col padded: [5.7, 4.5, 5.7, 4.5, 5.7]
    // Row 3 (wrap row 0):  [1.0, 1.2] -> Col padded: [1.2, 1.0, 1.2, 1.0, 1.2]
    var expected_array = [_][5]f32{
        [_]f32{ 5.7, 4.5, 5.7, 4.5, 5.7 }, // Wrapped row 2, cols wrapped
        [_]f32{ 1.2, 1.0, 1.2, 1.0, 1.2 }, // Data row 0, cols wrapped
        [_]f32{ 3.4, 2.3, 3.4, 2.3, 3.4 }, // Data row 1, cols wrapped
        [_]f32{ 5.7, 4.5, 5.7, 4.5, 5.7 }, // Data row 2, cols wrapped
        [_]f32{ 1.2, 1.0, 1.2, 1.0, 1.2 }, // Wrapped row 0, cols wrapped
    };
    var expected_shape = [_]usize{ 5, 5 };
    var expected_tensor = try Tensor(f32).fromArray(&allocator, &expected_array, &expected_shape);
    defer expected_tensor.deinit();

    try expectTensorEqual(f32, &expected_tensor, &output);
}

// Test Case 6: Padding a 3D tensor with constant mode
test "pads constant mode 3D" {
    tests_log.info("\n     test: pads constant mode 3D", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_][2][2]u8{
        [_][2]u8{
            [_]u8{ 1, 2 },
            [_]u8{ 3, 4 },
        },
        [_][2]u8{
            [_]u8{ 5, 6 },
            [_]u8{ 7, 8 },
        },
    };
    var data_shape = [_]usize{ 2, 2, 2 };
    var data_tensor = try Tensor(u8).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    // Pad 1 element start/end on axis 1, 1 element start on axis 2
    // pads = [p0s, p1s, p2s, p0e, p1e, p2e]
    var pads_array = [_]i64{ 0, 1, 1, 0, 1, 0 };
    var pads_shape = [_]usize{6};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var output = try TensMath.pads(u8, &data_tensor, &pads_tensor, "constant", @as(u8, 99), null);
    defer output.deinit();

    // Expected shape: [2, 2+1+1, 2+1+0] = [2, 4, 3]
    var expected_array = [_][4][3]u8{
        [_][3]u8{ // Batch 0
            [_]u8{ 99, 99, 99 }, // Padded row
            [_]u8{ 99, 1, 2 }, // Original [1, 2] with pad start
            [_]u8{ 99, 3, 4 }, // Original [3, 4] with pad start
            [_]u8{ 99, 99, 99 }, // Padded row
        },
        [_][3]u8{ // Batch 1
            [_]u8{ 99, 99, 99 }, // Padded row
            [_]u8{ 99, 5, 6 }, // Original [5, 6] with pad start
            [_]u8{ 99, 7, 8 }, // Original [7, 8] with pad start
            [_]u8{ 99, 99, 99 }, // Padded row
        },
    };
    var expected_shape = [_]usize{ 2, 4, 3 };
    var expected_tensor = try Tensor(u8).fromArray(&allocator, &expected_array, &expected_shape);
    defer expected_tensor.deinit();

    try expectTensorEqual(u8, &expected_tensor, &output);
}

// Test Case 7: Zero padding (should be identity)
test "pads zero padding" {
    tests_log.info("\n     test: pads zero padding", .{});
    const allocator = pkgAllocator.allocator;

    var data_array = [_][2]f32{
        [_]f32{ 1.0, 1.2 },
        [_]f32{ 2.3, 3.4 },
    };
    var data_shape = [_]usize{ 2, 2 };
    var data_tensor = try Tensor(f32).fromArray(&allocator, &data_array, &data_shape);
    defer data_tensor.deinit();

    var pads_array = [_]i64{ 0, 0, 0, 0 }; // No padding
    var pads_shape = [_]usize{4};
    var pads_tensor = try Tensor(i64).fromArray(&allocator, &pads_array, &pads_shape);
    defer pads_tensor.deinit();

    var output = try TensMath.pads(f32, &data_tensor, &pads_tensor, "constant", @as(f32, 99.0), null);
    defer output.deinit();

    // Expected should be the same as input
    try expectTensorEqual(f32, &data_tensor, &output);
}
