const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const TensorMath = zant.core.tensor.math_standard;
const ActivType = zant.model.layer.ActivationType;
const pkgAllocator = zant.utils.allocator;

test "tests description" {
    std.debug.print("\n--- Running activation_function tests\n", .{});
}

//*********************************************** ReLU ***********************************************
test "ReLU from ActivationFunction()" {
    std.debug.print("\n     test: ReLU from ActivationFunction()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_out = try TensorMath.ReLU(f32, &t1);
    defer t_out.deinit();

    try std.testing.expect(0.0 < t_out.data[0]);
    try std.testing.expect(0.0 == t_out.data[1]);
    try std.testing.expect(0.0 == t_out.data[2]);
    try std.testing.expect(0.0 < t_out.data[3]);
}

test "ReLU all negative" {
    std.debug.print("\n     test: ReLU all negative", .{});

    const allocator = std.testing.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_out = try TensorMath.ReLU(f32, &t1);
    defer t_out.deinit();

    for (t_out.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all positive" {
    std.debug.print("\n     test: ReLU all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_out = try TensorMath.ReLU(f32, &t1);
    defer t_out.deinit();

    for (t_out.data) |*val| {
        try std.testing.expect(val.* >= 0);
    }
}

test "ReLU backward " {
    std.debug.print("\n     test: ReLU all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t_input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t_input.deinit();

    var t_output = try TensorMath.ReLU(f32, &t_input);
    defer t_output.deinit();

    for (t_output.data) |*val| {
        try std.testing.expect(val.* >= 0);
    }

    // test backward----------------------------------------
    var dValues: [2][2]f32 = [_][2]f32{
        [_]f32{ 10.0, -20.0 },
        [_]f32{ -40.0, -50.0 },
    };

    var shape_dValues: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t_dValues = try Tensor(f32).fromArray(&allocator, &dValues, &shape_dValues);
    defer t_dValues.deinit();

    try TensorMath.ReLU_backward(f32, &t_dValues, &t_input);

    try std.testing.expect(10.0 == t_dValues.data[0]);
    try std.testing.expect(0.0 == t_dValues.data[1]);
    try std.testing.expect(0.0 == t_dValues.data[2]);
    try std.testing.expect(-50.0 == t_dValues.data[3]);
}

//*********************************************** Softmax ***********************************************

test "Softmax all positive" {
    std.debug.print("\n     test: Softmax all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.softmax(f32, &t1);
    defer t_output.deinit();

    //t_output.info();

    try std.testing.expect(t_output.data[0] + t_output.data[1] > 0.9);
    try std.testing.expect(t_output.data[0] + t_output.data[1] < 1.1);

    try std.testing.expect(t_output.data[2] + t_output.data[3] > 0.9);
    try std.testing.expect(t_output.data[2] + t_output.data[3] < 1.1);
}

test "Softmax all 0" {
    std.debug.print("\n     test: Softmax all 0", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.softmax(f32, &t1);
    defer t_output.deinit();

    //t1.info();

    try std.testing.expect(t_output.data[0] == t_output.data[1]);
    try std.testing.expect(t_output.data[2] == t_output.data[3]);
}

test "Softmax derivate" {
    std.debug.print("\n     test: Softmax derivate", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.softmax(f32, &t1);
    defer t_output.deinit();
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }
    //t1.info();

    try std.testing.expect(t_output.data[0] + t_output.data[1] > 0.9);
    try std.testing.expect(t_output.data[0] + t_output.data[1] < 1.1);

    try std.testing.expect(t_output.data[2] + t_output.data[3] > 0.9);
    try std.testing.expect(t_output.data[2] + t_output.data[3] < 1.1);

    try TensorMath.softmax_backward(f32, &t_output, &t1);
}

//*********************************************** Sigmoid ***********************************************

test "Sigmoid forward" {
    std.debug.print("\n     test: Sigmoid forward ", .{});

    var allocator = pkgAllocator.allocator;

    const input_data = [_]f64{ 0.0, 2.0, -2.0 }; // input data for the tensor
    var shape: [1]usize = [_]usize{3};
    var input_tensor = try Tensor(f64).fromArray(&allocator, &input_data, &shape); // create tensor from input data
    defer input_tensor.deinit();

    var t_output = try TensorMath.sigmoid(f64, &input_tensor);
    defer t_output.deinit();

    const expected_forward_output = [_]f64{ 0.5, 0.880797, 0.119203 }; // expected sigmoid output for each input value
    for (t_output.data, 0..) |*data, i| {
        try std.testing.expect(@abs(data.* - expected_forward_output[i]) < 1e-6);
    }
}

test "Sigmoid derivate" {
    std.debug.print("\n     test: Sigmoid derivate ", .{});

    const allocator = pkgAllocator.allocator;

    // Setup the gradient and act_forward_out tensors
    var gradient_data = [_]f64{ 0.2, 0.4, 0.6, 0.8 };
    var shape_grad: [1]usize = [_]usize{4};

    var act_forward_out_data = [_]f64{ 0.5, 0.7, 0.3, 0.9 };
    var shape_forw: [1]usize = [_]usize{4};

    var gradient_tensor = try Tensor(f64).fromArray(&allocator, &gradient_data, &shape_grad);
    defer gradient_tensor.deinit();
    var act_forward_out_tensor = try Tensor(f64).fromArray(&allocator, &act_forward_out_data, &shape_forw);
    defer act_forward_out_tensor.deinit();

    // Call the backward function
    try TensorMath.sigmoid_backward(f64, &gradient_tensor, &act_forward_out_tensor);

    // Expected values after applying the derivative
    const expected_values = [_]f64{
        0.2 * 0.5 * (1.0 - 0.5),
        0.4 * 0.7 * (1.0 - 0.7),
        0.6 * 0.3 * (1.0 - 0.3),
        0.8 * 0.9 * (1.0 - 0.9),
    };

    // Verify the result
    for (0..gradient_tensor.data.len) |i| {
        try std.testing.expect(gradient_tensor.data[i] - expected_values[i] < 0.0001);
    }
}

//*********************************************** LeakyReLU ***********************************************

test "LeakyReLU from ActivationFunction()" {
    std.debug.print("\n     test: LeakyReLU from ActivationFunction()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.leakyReLU(f32, &t1, 0.01);
    defer t_output.deinit();

    try std.testing.expect(t_output.data[0] == 1.0);
    try std.testing.expect(@abs(t_output.data[1] - (-0.02)) < 0.00001);
    try std.testing.expect(@abs(t_output.data[2] - (-0.04)) < 0.00001);
    try std.testing.expect(t_output.data[3] == 5.0);
}

test "LeakyReLU all negative" {
    std.debug.print("\n     test: LeakyReLU all negative", .{});

    const allocator = std.testing.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.leakyReLU(f32, &t1, 0.01);
    defer t_output.deinit();

    for (0..t_output.data.len) |i| {
        try std.testing.expect(@abs(t_output.data[i] - (0.01 * inputArray[i / shape[1]][i % shape[1]])) < 0.00001);
    }
}

test "LeakyReLU all positive" {
    std.debug.print("\n     test: LeakyReLU all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var t_output = try TensorMath.leakyReLU(f32, &t1, 0.01);
    defer t_output.deinit();

    for (0..t_output.data.len) |i| {
        try std.testing.expect(t_output.data[i] == inputArray[i / shape[1]][i % shape[1]]);
    }
}

test "LeakyReLU backward" {
    std.debug.print("\n     test: LeakyReLU backward", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t_input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t_input.deinit();

    var t_output = try TensorMath.leakyReLU(f32, &t_input, 0.01);
    defer t_output.deinit();

    var dValues: [2][2]f32 = [_][2]f32{
        [_]f32{ 10.0, -20.0 },
        [_]f32{ -40.0, -50.0 },
    };

    var t_dValues = try Tensor(f32).fromArray(&allocator, &dValues, &shape);
    defer t_dValues.deinit();

    try TensorMath.leakyReLU_backward(f32, &t_dValues, &t_input, 0.01);

    try std.testing.expect(t_dValues.data[0] == 10.0);
    try std.testing.expect(@abs(t_dValues.data[1] - (-20.0 * 0.01)) < 0.00001);
    try std.testing.expect(@abs(t_dValues.data[2] - (-40.0 * 0.01)) < 0.00001);
    try std.testing.expect(t_dValues.data[3] == -50.0);
}
