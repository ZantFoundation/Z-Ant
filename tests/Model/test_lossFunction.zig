const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const TensMath = zant.core.tensor.math_standard;

const Loss = zant.model.loss_function;
const LossType =Loss.LossType;
const MSELoss =Loss.MSELoss;
const CCELoss =Loss.CCELoss;
const error_handler = zant.utils.error_handler;
const LossError = error_handler.LossError;
const TensorError = error_handler.TensorError;
const pkgAllocator = zant.utils.allocator;

test "tests description" {
    std.debug.print("\n--- Running loss_function tests\n", .{});
}

// INTERFACE LOSS FUNCTION TESTS--------------------------------------------------------------------------------------------------

test " Loss Function MSE using Interface, target==predictor" {
    std.debug.print("\n     test: Loss Function MSE using Interface, target==predictor ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //LOSS SHOULD RESULT ALL ZEROS
    const mse = Loss.LossFunction(LossType.MSE){};
    var loss: Tensor(f32) = try mse.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " Loss Function CCE using Interface, target==predictor" {
    std.debug.print("\n     test: Loss Function CCE using Interface, target==predictor", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    const cce = Loss.LossFunction(LossType.CCE){};
    var loss: Tensor(f32) = try cce.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    // CCE should give very small values when target equals prediction
    const epsilon = 1e-10;
    for (0..loss.size) |i| {
        try std.testing.expect(loss.data[i] < epsilon);
    }
}

// LOSS FUNCTION TESTS--------------------------------------------------------------------------------------------------

test " MSE target==predictor, 2 x 2" {
    std.debug.print("\n     test: MSE target==predictor, 2 x 2 ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //LOSS SHOULD RESULT ALL ZEROS
    const mse = Loss.LossFunction(LossType.MSE){};
    var loss: Tensor(f32) = try mse.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " MSE target==predictor, 2 x 3 X 2" {
    std.debug.print("\n     test: MSE target==predictor, 2 x 3 X 2 ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][2]u32 = [_][3][2]u32{
        [_][2]u32{
            [_]u32{ 1, 2 },
            [_]u32{ 4, 5 },
            [_]u32{ 6, 7 },
        },
        [_][2]u32{
            [_]u32{ 10, 20 },
            [_]u32{ 40, 50 },
            [_]u32{ 60, 70 },
        },
    };

    var shape: [3]usize = [_]usize{ 2, 3, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(u32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(u32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //std.debug.print("\n creating a new MSE loss function", .{});

    //LOSS SHOULD RESULT ALL ZEROS
    const mse = Loss.LossFunction(LossType.MSE){};
    var loss: Tensor(u32) = try mse.computeLoss(u32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " MSE target!=predictor, 2 x 3 X 2" {
    std.debug.print("\n     test: MSE target!=predictor, 2 x 3 X 2", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][2]i32 = [_][3][2]i32{
        [_][2]i32{
            [_]i32{ 1, 2 },
            [_]i32{ 4, 5 },
            [_]i32{ 6, 7 },
        },
        [_][2]i32{
            [_]i32{ 10, 20 },
            [_]i32{ 40, 50 },
            [_]i32{ 60, 70 },
        },
    };

    var inputArray2: [2][3][2]i32 = [_][3][2]i32{
        [_][2]i32{
            [_]i32{ 10, 20 },
            [_]i32{ 40, 50 },
            [_]i32{ 60, 70 },
        },
        [_][2]i32{
            [_]i32{ 1, 2 },
            [_]i32{ 4, 5 },
            [_]i32{ 6, 7 },
        },
    };

    var shape: [3]usize = [_]usize{ 2, 3, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(i32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(i32).fromArray(&allocator, &inputArray2, &shape);
    defer t2_PREDICTION.deinit();

    //std.debug.print("\n creating a new MSE loss function", .{});

    //LOSS SHOULD RESULT ALL ZEROS
    const mse = Loss.LossFunction(LossType.MSE){};
    var loss: Tensor(i32) = try mse.computeLoss(i32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 != loss.data[i]);
    }
}

test " CCE target==predictor, 2 x 2 all 1" {
    std.debug.print("\n     test:CCE target==predictor, 2 x 2", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 1.0 },
        [_]f32{ 1.0, 1.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    const cce = Loss.LossFunction(LossType.CCE){};
    var loss: Tensor(f32) = try cce.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    //loss.info();
}

// GRADIENT TESTS-------------------------------------------------------------------------------------------------------

test " GRADIENT MSE target==predictor, 2 x 2" {
    std.debug.print("\n     test: GRADIENT MSE target==predictor, 2 x 2 ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //gradient SHOULD RESULT ALL ZEROS
    const mse = Loss.LossFunction(LossType.MSE){};
    var Grad = try mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET);
    defer Grad.deinit();
}

test " GRADIENT MSE error on shape (dimensions)" {
    std.debug.print("\n     test: GRADIENT MSE error on shape (dimensions)", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape2);
    defer t2_PREDICTION.deinit();

    const mse = Loss.LossFunction(LossType.MSE){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET));
}

test " GRADIENT MSE error on shape (len)" {
    std.debug.print("\n     test: GRADIENT MSE error on shape (len)", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 4 }; // 2x2 matrix
    var shape2: [3]usize = [_]usize{ 2, 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromShape(&allocator, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t2_PREDICTION.deinit();

    const mse = Loss.LossFunction(LossType.MSE){};
    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET));
}

test " GRADIENT CCE error on shape (dimensions)" {
    std.debug.print("\n     test: GRADIENT CCE error on shape (dimensions)", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape2);
    defer t2_PREDICTION.deinit();

    const cce = Loss.LossFunction(LossType.CCE){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, cce.computeGradient(f32, &t2_PREDICTION, &t1_TARGET));
}

test " GRADIENT CCE error on shape (len)" {
    std.debug.print("\n     test: GRADIENT CCE error on shape (len)", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 4 }; // 2x2 matrix
    var shape2: [3]usize = [_]usize{ 2, 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromShape(&allocator, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t2_PREDICTION.deinit();
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape1);
    defer t3_GRAD.deinit();

    const cce = Loss.LossFunction(LossType.CCE){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, cce.computeGradient(f32, &t2_PREDICTION, &t1_TARGET));
}

//LIMIT CASES--------------------------------------------------

test "empty vector" {
    std.debug.print("\n     test: GRADIENT CCE error on shape (len)", .{});
    const allocator = pkgAllocator.allocator;

    var t1_TARGET = try Tensor(f32).init(&allocator);
    defer t1_TARGET.deinit();

    var t2_PREDICTION = try Tensor(f32).init(&allocator);
    defer t2_PREDICTION.deinit();

    var t3_GRAD = try Tensor(f64).init(&allocator);
    defer t3_GRAD.deinit();

    const cce = Loss.LossFunction(LossType.CCE){};

    try std.testing.expectError(TensorError.EmptyTensor, cce.computeGradient(f32, &t2_PREDICTION, &t1_TARGET));
}
