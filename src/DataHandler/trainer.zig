//! This file contains at the moment all the available options to train a model.
//! Dependng on your intentions you can use trainTensors(), a general trainer for tensors,
//! or TrainDataLoader(), more specific for training data loaded from a file. This last one has been well tested for MNIST.

const std = @import("std");

const zant = @import("../zant.zig");

const Tensor = zant.core.tensor;
const TensMath = zant.core.tensor.math_standard;
const Model = zant.model.Model;

const Loss = zant.model.loss_function;
const LossType = Loss.LossType;
const Optim = zant.model.optim;

const DataLoader = zant.data_handler.data_loader.DataLoader;
const DataProc = zant.data_handler.data_processor;
const NormalizType = zant.data_handler.data_processor.NormalizationType;

const DenseLayer = zant.model.layer.DenseLayer;
const ConvolutionalLayer = zant.model.layer.ConvolutionalLayer;

/// Defines the type of trainer used for model training.
///
/// - `DataLoaderTrainer`: Uses a `DataLoader` to feed batches of data into the model during training.
/// - `TensorTrainer`: Uses direct tensor inputs for training.
pub const TrainerType = enum {
    DataLoaderTrainer,
    TensorTrainer,
};

pub fn TrainDataLoader(
    comptime T: type,
    comptime XType: type, // Input types
    comptime YType: type, // Output type
    comptime allocator: *const std.mem.Allocator,
    comptime batchSize: i16,
    features: usize,
    model: *Model(T, allocator),
    load: *DataLoader(T, XType, YType, batchSize),
    epochs: u32,
    comptime lossType: LossType,
    comptime lr: f64,
    training_size: f32,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);

    var AccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(AccuracyRecord);

    var ValidationLossRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationLossRecord);

    var ValidationAccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationAccuracyRecord);

    var shapeXArr = [_]usize{ batchSize, features };
    var shapeYArr = [_]usize{batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    var steps: u16 = 0;
    try load.trainTestSplit(allocator, training_size);

    const train_len: u16 = @as(u16, @intCast(load.X_train.?.len));
    steps = @divFloor(train_len, batchSize);
    if (train_len % batchSize != 0) {
        steps += 1;
    }

    std.debug.print("Number of training steps: {}\n", .{steps});

    for (0..epochs) |i| {
        var totalCorrect: u16 = 0;
        var totalSamples: u16 = 0;

        var totalCorrectVal: u16 = 0;
        var totalSamplesVal: u16 = 0;

        var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){};

        for (0..steps) |step| {
            _ = load.xTrainNextBatch();
            _ = load.yTrainNextBatch();
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            const reshaped = try allocator.create(Tensor.Tensor(T));
            reshaped.* = try TensMath.reshape(T, predictions, &shape, null);
            predictions.deinit();
            predictions = reshaped;

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;

            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, predictions, &load.yTensor);
            defer grad.deinit();

            _ = try model.backward(&grad);

            try optimizer.step(model);

            std.debug.print("Training - Epoch: {}, Step: {}, Loss: {}, Accuracy: {} \n", .{ i + 1, step + 1, LossMeanRecord[i], AccuracyRecord[i] });
        }

        load.reset();

        const val_len: u16 = @as(u16, @intCast(load.X_test.?.len));
        var val_steps: u16 = @divFloor(val_len, batchSize);
        if (val_len % batchSize != 0) {
            val_steps += 1;
        }

        std.debug.print("\nNumber of validation steps: {}\n", .{val_steps});

        for (0..val_steps) |step| {
            _ = load.xTestNextBatch(batchSize);
            _ = load.yTestNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            const reshaped = try allocator.create(Tensor.Tensor(T));
            reshaped.* = try TensMath.reshape(T, predictions, &shape, null);
            predictions.deinit();
            predictions = reshaped;

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, predictions, &load.yTensor);
            totalCorrectVal += correctPredictions;
            totalSamplesVal += batchSize;

            ValidationLossRecord[i] = TensMath.mean(T, &loss);
            ValidationAccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrectVal)) / @as(f32, @floatFromInt(totalSamplesVal)) * 100.0;

            std.debug.print("\nValidation - Epoch: {}, Step: {}", .{ i + 1, step + 1 });
        }

        load.reset();

        std.debug.print("\nEpoch {}: Training Loss = {}, Training Accuracy = {}%", .{ i + 1, LossMeanRecord[i], AccuracyRecord[i] });
        std.debug.print("\nEpoch {}: Validation Loss = {}, Validation Accuracy = {}%", .{ i + 1, ValidationLossRecord[i], ValidationAccuracyRecord[i] });
    }
}

pub fn TrainDataLoader2D(
    comptime T: type,
    comptime XType: type,
    comptime YType: type,
    allocator: *const std.mem.Allocator,
    comptime batchSize: i16,
    features: usize,
    model: *Model(T),
    load: *DataLoader(T, XType, YType, batchSize, 3),
    epochs: u32,
    comptime lossType: LossType,
    comptime lr: f64,
    training_size: f32,
    l2_lambda: T,
    max_grad_norm: T,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);

    var AccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(AccuracyRecord);

    var ValidationLossRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationLossRecord);

    var ValidationAccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationAccuracyRecord);

    _ = features;
    var shapeXArr = [_]usize{ batchSize, 1, 28, 28 };
    var shapeYArr = [_]usize{batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    var steps: u16 = 0;
    try load.trainTestSplit(allocator, training_size);

    const train_len: u16 = @as(u16, @intCast(load.X_train.?.len));
    steps = @divFloor(train_len, batchSize);
    if (train_len % batchSize != 0) {
        steps += 1;
    }

    print_start_training();
    std.debug.print("Number of training steps: {}\n", .{steps});

    for (0..epochs) |i| {
        var totalCorrect: u16 = 0;
        var totalSamples: u16 = 0;

        var totalCorrectVal: u16 = 0;
        var totalSamplesVal: u16 = 0;

        for (0..steps) |step| {
            _ = load.xTrainNextBatch();
            _ = load.yTrainNextBatch();
            try load.toTensor(allocator, &shapeX, &shapeY);

            try convertToOneHot(T, batchSize, &load.yTensor);
            try DataProc.normalize(T, &load.xTensor, NormalizType.UnityBasedNormalizartion);
            var predictions = try model.forward(&load.xTensor);
            //predictions.print();
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            const reshaped = try allocator.create(Tensor.Tensor(T));
            reshaped.* = try TensMath.reshape(T, predictions, &shape, null);
            predictions.deinit();
            predictions = reshaped;
            //predictions.print();
            // DEBUG try predictions.isSafe();

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;
            std.debug.print("\nLOSS: {d}\n", .{LossMeanRecord[i]});
            std.debug.print("\nACCURACY: {d}\n", .{AccuracyRecord[i]});
            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, predictions, &load.yTensor);
            defer grad.deinit();

            // Apply L2 and gradient clipping before backward
            for (model.layers.items) |layer_| {
                if (layer_.layer_type == .DenseLayer) {
                    const dense_layer = @as(*DenseLayer(T), @alignCast(@ptrCast(layer_.layer_ptr)));
                    // Apply L2 regularization
                    for (dense_layer.w_gradients.data, dense_layer.weights.data) |*grad_w, weight| {
                        grad_w.* += l2_lambda * weight;
                    }
                    // Clip gradients
                    try clipGradients(T, &dense_layer.w_gradients, max_grad_norm);
                    try clipGradients(T, &dense_layer.b_gradients, max_grad_norm);
                } else if (layer_.layer_type == .ConvolutionalLayer) {
                    const conv_layer = @as(*ConvolutionalLayer(T), @alignCast(@ptrCast(layer_.layer_ptr)));
                    // Apply L2 regularization
                    for (conv_layer.w_gradients.data, conv_layer.weights.data) |*grad_w, weight| {
                        grad_w.* += l2_lambda * weight;
                    }
                    // Clip gradients
                    try clipGradients(T, &conv_layer.w_gradients, max_grad_norm);
                    try clipGradients(T, &conv_layer.b_gradients, max_grad_norm);
                }
            }

            _ = try model.backward(&grad);

            var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr){};
            try optimizer.step(model);

            std.debug.print("Training - Epoch: {}, Step: {}, Loss: {}, Accuracy: {} \n", .{ i + 1, step + 1, LossMeanRecord[i], AccuracyRecord[i] });
        }

        load.reset();

        const val_len: u16 = @as(u16, @intCast(load.X_test.?.len));
        var val_steps: u16 = @divFloor(val_len, batchSize);
        if (val_len % batchSize != 0) {
            val_steps += 1;
        }

        std.debug.print("\nNumber of validation steps: {}\n", .{val_steps});

        for (0..val_steps) |step| {
            _ = load.xTestNextBatch();
            _ = load.yTestNextBatch();
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);
            //try DataProc.normalize(T, &load.xTensor, NormalizType.UnityBasedNormalizartion);
            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            const reshaped = try allocator.create(Tensor.Tensor(T));
            reshaped.* = try TensMath.reshape(T, predictions, &shape, null);
            predictions.deinit();
            predictions = reshaped;

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, predictions, &load.yTensor);
            totalCorrectVal += correctPredictions;
            totalSamplesVal += batchSize;

            ValidationLossRecord[i] = TensMath.mean(T, &loss);
            ValidationAccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrectVal)) / @as(f32, @floatFromInt(totalSamplesVal)) * 100.0;

            std.debug.print("\nValidation - Epoch: {}, Step: {}", .{ i + 1, step + 1 });
        }

        load.reset();

        std.debug.print("\nEpoch {}: Training Loss = {}, Training Accuracy = {}%", .{ i + 1, LossMeanRecord[i], AccuracyRecord[i] });
        std.debug.print("\nEpoch {}: Validation Loss = {}, Validation Accuracy = {}%", .{ i + 1, ValidationLossRecord[i], ValidationAccuracyRecord[i] });
    }

    print_end_training();
}

/// Computes the accuracy of model predictions by comparing predicted and actual labels.
fn computeAccuracy(comptime T: type, predictions: *Tensor.Tensor(T), targets: *Tensor.Tensor(T)) !u16 {
    var correct: u16 = 0;
    const rows = predictions.shape[0];
    const cols = predictions.shape[1];

    for (0..rows) |i| {
        var predictedLabel: usize = 0;
        var maxVal: T = predictions.data[i * cols];

        // Find the class with the highest value in predictions
        for (1..cols) |j| {
            const val = predictions.data[i * cols + j];
            if (val > maxVal) {
                maxVal = val;
                predictedLabel = j;
            }
        }

        // Find the actual label
        var actualLabel: usize = 0;
        var maxTargetVal: T = targets.data[i * cols];
        for (1..cols) |j| {
            const val = targets.data[i * cols + j];
            if (val > maxTargetVal) {
                maxTargetVal = val;
                actualLabel = j;
            }
        }

        // Check if the prediction is correct
        if (predictedLabel == actualLabel) {
            correct += 1;
        }
    }

    return correct;
}

fn convertToOneHot(comptime T: type, batchSize: i16, yBatch: *Tensor.Tensor(T)) !void {
    const numClasses = 10;

    var shapeYArr = [_]usize{ @intCast(batchSize), numClasses };
    const oneHotShape = &shapeYArr;

    var oneHotYBatch = try Tensor.Tensor(T).fromShape(yBatch.allocator, oneHotShape);

    for (0..@intCast(batchSize)) |i| {
        const label: usize = (@intFromFloat(yBatch.data[i]));
        for (0..numClasses) |j| {
            if (j == label) {
                oneHotYBatch.data[i * numClasses + j] = 1;
            } else {
                oneHotYBatch.data[i * numClasses + j] = 0;
            }
        }
    }

    yBatch.deinit();
    yBatch.* = oneHotYBatch;
}

pub fn trainTensors(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    model: *Model(T),
    input: *Tensor.Tensor(T),
    targets: *Tensor.Tensor(T),
    epochs: u32,
    comptime lr: f64,
) !void {
    const loser = Loss.LossFunction(LossType.MSE){};
    var optimizer = Optim.Optimizer(T, T, T, Optim.optimizer_SGD, lr){};

    // allocate only once
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);
    var predictions: ?*Tensor.Tensor(T) = null; //it gets already free by model.deinit() and gets free at each iteration except the first

    var loss: ?Tensor.Tensor(T) = null;
    defer loss.?.deinit();
    var grad: ?Tensor.Tensor(T) = null;
    defer grad.?.deinit();

    for (0..epochs) |i| {
        std.debug.print("\n\n----------------------epoch:{}", .{i});

        // Forward pass
        std.debug.print("\n-------------------------------forwarding", .{});

        if (predictions != null) predictions.?.deinit();
        const forward_result = try model.forward(input);
        predictions = forward_result;

        // Loss computation
        std.debug.print("\n-------------------------------computing loss", .{});
        if (loss != null) loss.?.deinit();
        loss = try loser.computeLoss(T, predictions.?, targets);

        LossMeanRecord[i] = TensMath.mean(T, &loss.?);
        std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

        // Gradient computation
        std.debug.print("\n-------------------------------computing loss gradient", .{});
        if (grad != null) grad.?.deinit();
        grad = try loser.computeGradient(T, predictions.?, targets);
        std.debug.print("\n     gradient:", .{});

        // Backpropagation
        std.debug.print("\n-------------------------------backwarding", .{});
        try model.backward(&grad.?);

        // Optimization
        std.debug.print("\n-------------------------------Optimizer Step", .{});
        try optimizer.step(model);
    }

    predictions.?.*.deinit();

    std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
}

fn print_start_training() void {
    const str =
        \\
        \\   _____ __             __     __             _       _                   
        \\  / ___// /_____ ______/ /_   / /__________ _(_)___  (_)___  ____ _       
        \\  \__ \/ __/ __ `/ ___/ __/  / __/ ___/ __ `/ / __ \/ / __ \/ __ `/       
        \\ ___/ / /_/ /_/ / /  / /_   / /_/ /  / /_/ / / / / / / / / / /_/ /  _ _ _ 
        \\/____/\__/\__,_/_/   \__/   \__/_/   \__,_/_/_/ /_/_/_/ /_/\__, /  (_|_|_)
        \\                                                          /____/          
        \\ 
    ;

    std.debug.print("{s}", .{str});
}

fn print_end_training() void {
    const str =
        \\
        \\    ______          __   __             _       _                   
        \\   / ____/___  ____/ /  / /__________ _(_)___  (_)___  ____ _       
        \\  / __/ / __ \/ __  /  / __/ ___/ __ `/ / __ \/ / __ \/ __ `/       
        \\ / /___/ / / / /_/ /  / /_/ /  / /_/ / / / / / / / /_/ /  _ _ _ 
        \\/_____/_/ /_/\__,_/   \__/_/   \__,_/_/_/ /_/_/_/ /_/\__, /  (_|_|_)
        \\                                                 /____/                 
        \\ 
    ;

    std.debug.print("{s}", .{str});
}

// Helper function for gradient clipping
fn clipGradients(comptime T: type, gradients: *Tensor.Tensor(T), max_norm: T) !void {
    var total_norm: T = 0;

    for (gradients.data) |grad| {
        total_norm += grad * grad;
    }
    total_norm = @sqrt(total_norm);

    if (total_norm > max_norm) {
        const scaling_factor = max_norm / (total_norm + 1e-6);
        for (gradients.data) |*grad| {
            grad.* *= scaling_factor;
        }
    }
}
