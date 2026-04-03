const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.lib_elementWise);

test "Sum two tensors on CPU architecture" {
    tests_log.info("\n     test: Sum two tensors on CPU architecture", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    var t3 = try TensMath.sum_tensors(f32, f64, &t1, &t2); // Output tensor with larger type
    defer t3.deinit();

    // Check if the values in t3 are as expected
    try std.testing.expect(2.0 == t3.data[0]);
    try std.testing.expect(4.0 == t3.data[1]);
}

test "test tensor element-wise multiplication" {
    tests_log.info("\n     test: tensor element-wise multiplication ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.mul(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] * tensor2.data[i]);
    }
}

test "Error when input tensors have different sizes" {
    tests_log.info("\n     test: Error when input tensors have different sizes", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 14.0, 15.0 },
    };

    var shape1 = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2 = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    try std.testing.expectError(TensorMathError.IncompatibleBroadcastShapes, TensMath.sum_tensors(f32, f64, &t1, &t2));
}

test "add bias" {
    tests_log.info("\n     test:add bias", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    const flatArr: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var shape_bias: [1]usize = [_]usize{3};
    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &t1, &bias);

    for (t1.data, 0..) |*data, i| {
        try std.testing.expect(data.* == flatArr[i] + 1);
    }

    t1.deinit();
    bias.deinit();
}

test "Subtraction with same shape tensors" {
    tests_log.info("\n     test: Subtraction with same shape tensors", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 6.0);
    try std.testing.expectEqual(result.data[3], 7.0);
}

test "Subtraction with broadcasting - scalar and matrix" {
    tests_log.info("\n     test: Subtraction with broadcasting - scalar and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Scalar (1x1)
    var inputArray2 = [_]f32{2.0};
    var shape2: [1]usize = [_]usize{1};

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 3.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 7.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with broadcasting - row and matrix" {
    tests_log.info("\n     test: Subtraction with broadcasting - row and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Row vector as 2D array with broadcasting shape
    var inputArray2: [1][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
    };
    var shape2: [2]usize = [_]usize{ 1, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 8.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with incompatible shapes" {
    tests_log.info("\n     test: Subtraction with incompatible shapes", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Matrix 3x2 (incompatible)
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
        [_]f32{ 5.0, 6.0 },
    };
    var shape2: [2]usize = [_]usize{ 3, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    try std.testing.expectError(TensorMathError.IncompatibleBroadcastShapes, TensMath.sub_tensors(f32, f32, &t1, &t2));
}

test "Lean subtraction with SIMD optimization" {
    tests_log.info("\n     test: Lean subtraction with SIMD optimization", .{});
    const allocator = pkgAllocator.allocator;

    // Create larger tensors to test SIMD optimization
    var inputArray1: [4][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 3.0, 4.0 },
        [_]f32{ 5.0, 6.0, 7.0, 8.0 },
        [_]f32{ 9.0, 10.0, 11.0, 12.0 },
        [_]f32{ 13.0, 14.0, 15.0, 16.0 },
    };

    var inputArray2: [4][4]f32 = [_][4]f32{
        [_]f32{ 0.5, 1.0, 1.5, 2.0 },
        [_]f32{ 2.5, 3.0, 3.5, 4.0 },
        [_]f32{ 4.5, 5.0, 5.5, 6.0 },
        [_]f32{ 6.5, 7.0, 7.5, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 4, 4 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    // Test first few elements to verify SIMD optimization
    try std.testing.expectEqual(result.data[0], 0.5); // 1.0 - 0.5
    try std.testing.expectEqual(result.data[1], 1.0); // 2.0 - 1.0
    try std.testing.expectEqual(result.data[2], 1.5); // 3.0 - 1.5
    try std.testing.expectEqual(result.data[3], 2.0); // 4.0 - 2.0
    try std.testing.expectEqual(result.data[4], 2.5); // 5.0 - 2.5
    try std.testing.expectEqual(result.data[5], 3.0); // 6.0 - 3.0
    try std.testing.expectEqual(result.data[6], 3.5); // 7.0 - 3.5
    try std.testing.expectEqual(result.data[7], 4.0); // 8.0 - 4.0

    // Test last few elements
    try std.testing.expectEqual(result.data[14], 7.5); // 15.0 - 7.5
    try std.testing.expectEqual(result.data[15], 8.0); // 16.0 - 8.0
}

test "test tensor element-wise division" {
    tests_log.info("\n     test: tensor element-wise division ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 2, 4, 6 },
        [_]u8{ 8, 10, 12 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.div(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] / tensor2.data[i]);
    }
}

test "Sum list of tensors - normal case" {
    tests_log.info("\n     test: Sum list of tensors - normal case", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };
    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };
    var inputArray3: [2][2]f32 = [_][2]f32{
        [_]f32{ 9.0, 10.0 },
        [_]f32{ 11.0, 12.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();
    var t3 = try Tensor(f32).fromArray(&allocator, &inputArray3, &shape);
    defer t3.deinit();

    var tensors = [_]*Tensor(f32){ &t1, &t2, &t3 };
    var result = try TensMath.sum_tensor_list(f32, f32, &tensors);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 15.0); // 1 + 5 + 9
    try std.testing.expectEqual(result.data[1], 18.0); // 2 + 6 + 10
    try std.testing.expectEqual(result.data[2], 21.0); // 3 + 7 + 11
    try std.testing.expectEqual(result.data[3], 24.0); // 4 + 8 + 12
}

test "Sum list of tensors - single tensor case" {
    tests_log.info("\n     test: Sum list of tensors - single tensor case", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var tensors = [_]*Tensor(f32){&t1};
    var result = try TensMath.sum_tensor_list(f32, f32, &tensors);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 1.0);
    try std.testing.expectEqual(result.data[1], 2.0);
    try std.testing.expectEqual(result.data[2], 3.0);
    try std.testing.expectEqual(result.data[3], 4.0);
}

test "Sum list of tensors - empty list error" {
    tests_log.info("\n     test: Sum list of tensors - empty list error", .{});

    var tensors = [_]*Tensor(f32){};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.sum_tensor_list(f32, f32, &tensors));
}

test "Sum list of tensors - different sizes error" {
    tests_log.info("\n     test: Sum list of tensors - different sizes error", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
        [_]f32{ 9.0, 10.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 };
    var shape2: [2]usize = [_]usize{ 3, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var tensors = [_]*Tensor(f32){ &t1, &t2 };
    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.sum_tensor_list(f32, f32, &tensors));
}

test "Sum list of tensors - type promotion" {
    tests_log.info("\n     test: Sum list of tensors - type promotion", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]u8 = [_][2]u8{
        [_]u8{ 100, 100 },
        [_]u8{ 100, 100 },
    };
    var inputArray2: [2][2]u8 = [_][2]u8{
        [_]u8{ 100, 100 },
        [_]u8{ 100, 100 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]*Tensor(u8){ &t1, &t2 };
    var result = try TensMath.sum_tensor_list(u8, u32, &tensors); // Promote to u32 to handle overflow
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 200);
    try std.testing.expectEqual(result.data[1], 200);
    try std.testing.expectEqual(result.data[2], 200);
    try std.testing.expectEqual(result.data[3], 200);
}

test "test tensor element-wise tanh with valid f32 tensor" {
    tests_log.info("\n     test: tensor element-wise tanh with valid f32 tensor", .{});
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 1.0, -1.0 },
        [_]f32{ 0.5, -0.5, 2.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var result = try TensMath.tanh(f32, &tensor);
    defer result.deinit();

    const epsilon: f32 = 1e-6;
    for (0..result.size) |i| {
        const expected = std.math.tanh(tensor.data[i]);
        try std.testing.expect(std.math.approxEqAbs(f32, result.data[i], expected, epsilon) == true);
    }
}

test "test tensor element-wise ceil with valid f32 tensor" {
    tests_log.info("\n     test: tensor element-wise ceil with valid f32 tensor", .{});
    const allocator = std.testing.allocator;

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 1.2, -1.2 },
        [_]f32{ 0.5, -0.5, 2.0 },
    };
    const expectedArray: [6]f32 = [_]f32{ 0.0, 2.0, -1.0, 1.0, 0.0, 2.0 };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var result = try TensMath.ceil(f32, &tensor);
    defer result.deinit();

    const epsilon: f32 = 1e-6;
    for (0..result.size) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, result.data[i], expectedArray[i], epsilon) == true);
    }
}

test "floor_standard - basic case with special values" {
    tests_log.info("\n     test: floor_standard - basic case with special values", .{});
    const allocator = pkgAllocator.allocator;

    // Input
    var input_array = [_]f32{ 1.7, -2.3, 3.0, 0.0, -0.0, std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    var shape = [_]usize{8};
    var input = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
    defer input.deinit();

    // Floor
    var result = try TensMath.floor(f32, &input);
    defer result.deinit();

    // Expected output
    const expected = [_]f32{ 1.0, -3.0, 3.0, 0.0, -0.0, std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    try std.testing.expectEqual(expected.len, result.data.len);
    for (expected, result.data) |exp, res| {
        if (std.math.isNan(exp)) {
            try std.testing.expect(std.math.isNan(res));
        } else {
            try std.testing.expectEqual(exp, res);
        }
    }
    try std.testing.expectEqualSlices(usize, &shape, result.shape);
}

test "QuantizeLinear basic with u8 output" {
    tests_log.info("\n     test: QuantizeLinear basic with uint8 output", .{});
    const allocator = std.testing.allocator;

    const input_array = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    const scale_val: f32 = 0.5;
    const zero_point_val: u8 = 128;

    var shape = [_]usize{4};
    var input = try Tensor(f32).fromArray(&allocator, &input_array, &shape);
    defer input.deinit();

    var shape1 = [_]usize{1};
    var scale_tensor = try Tensor(f32).fromArray(&allocator, &[_]f32{scale_val}, &shape1);
    defer scale_tensor.deinit();

    var shape2 = [_]usize{1};
    var zp_tensor = try Tensor(u8).fromArray(&allocator, &[_]u8{zero_point_val}, &shape2);
    defer zp_tensor.deinit();

    var output = try TensMath.quantizeLinear(f32, u8, u8, &input, &scale_tensor, &zp_tensor, 0, -1);
    defer output.deinit();

    const expected = [_]u8{ 128, 130, 132, 134 };
    try std.testing.expectEqualSlices(u8, &expected, output.data);
}
