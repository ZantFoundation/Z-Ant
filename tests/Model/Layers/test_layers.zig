const std = @import("std");
const zant = @import("zant");
const layer_ = zant.model.layer;
const Layer = layer_.Layer;
const DenseLayer = layer_.DenseLayer;
const ConvolutionalLayer = layer_.ConvolutionalLayer;
const FlattenLayer = layer_.FlattenLayer;
const ActivationLayer = layer_.ActivationLayer;
const BatchNormLayer = layer_.BatchNormLayer;
const PoolingLayer = layer_.PoolingLayer;
const PoolingType = layer_.poolingLayer.PoolingType;
const Tensor = zant.core.tensor.Tensor;
const TensorMath = zant.core.tensor.math_standard;
const error_handler = zant.utils.error_handler;
const LayerError = error_handler.LayerError;
const TensorError = error_handler.TensorError;
const TensorMathError = error_handler.TensorMathError;
const tensor = zant.core.tensor;
const ActivationType = zant.model.layer.ActivationType;
const pkg_allocator = zant.utils.allocator;

test "Layer test description" {
    std.debug.print("\n--- Running Layer tests\n", .{});
}

test "Rand n and zeros" {
    const allocator = &pkg_allocator.allocator;
    const randomArray = try layer_.randn(f32, allocator, 2, 2);
    defer allocator.free(randomArray);
    const zerosArray = try layer_.zeros(f32, allocator, 2, 2);
    defer allocator.free(zerosArray);

    //test dimension
    try std.testing.expectEqual(randomArray.len, 4);
    try std.testing.expectEqual(zerosArray.len, 4);

    //test values
    for (0..2) |i| {
        for (0..2) |j| {
            try std.testing.expect(randomArray[i * 2 + j] != 0.0);
            try std.testing.expect(zerosArray[i * 2 + j] == 0.0);
        }
    }
}

test "DenseLayer forward and backward test" {
    std.debug.print("\n     test: DenseLayer forward test and backward testx", .{});
    const allocator = &pkg_allocator.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    const layer1 = DenseLayer(f64).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 4,
        .n_neurons = 2,
    }));
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
        [_]f64{ 14.0, 15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    const output_tensor = try layer1.forward(&input_tensor);
    try std.testing.expectEqual(output_tensor.shape[0], 5);
    try std.testing.expectEqual(output_tensor.shape[1], 2);

    // Check that after forward, output does not contain zeros
    for (0..5) |i| {
        for (0..2) |j| {
            try std.testing.expect(output_tensor.data[i * 2 + j] != 0.0);
        }
    }

    // Test backward, create array with right dimensions and random values as gradients
    var gradArray: [5][2]f64 = [_][2]f64{
        [_]f64{ 0.1, 0.2 },
        [_]f64{ 0.3, 0.4 },
        [_]f64{ 0.5, 0.6 },
        [_]f64{ 0.7, 0.8 },
        [_]f64{ 0.9, 1.0 },
    };
    var gradShape: [2]usize = [_]usize{ 5, 2 };

    var grad = try tensor.Tensor(f64).fromArray(allocator, &gradArray, &gradShape);
    defer grad.deinit();

    var backward = try layer1.backward(&grad);
    defer backward.deinit();

    // Check that bias and gradients are valid (non-zero)
    var myDense: *DenseLayer(f64) = @ptrCast(@alignCast(layer1.layer_ptr));
    for (0..2) |i| {
        try std.testing.expect(try myDense.bias.get(i) != 0.0);
        for (0..4) |j| {
            try std.testing.expect(try myDense.w_gradients.get(i + j) != 0.0);
        }
    }
}

test "test getters " {
    std.debug.print("\n     test: getters ", .{});
    const allocator = &pkg_allocator.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    const layer1 = DenseLayer(f64).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 4,
        .n_neurons = 2,
    }));
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
        [_]f64{ 14.0, 15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    _ = try layer1.forward(&input_tensor);
    //defer output_tensor.deinit();

    // Test backward, create array with right dimensions and random values as gradients
    var gradArray: [5][2]f64 = [_][2]f64{
        [_]f64{ 0.1, 0.2 },
        [_]f64{ 0.3, 0.4 },
        [_]f64{ 0.5, 0.6 },
        [_]f64{ 0.7, 0.8 },
        [_]f64{ 0.9, 1.0 },
    };
    var gradShape: [2]usize = [_]usize{ 5, 2 };

    var grad = try tensor.Tensor(f64).fromArray(allocator, &gradArray, &gradShape);
    defer grad.deinit();

    var backward = try layer1.backward(&grad);
    defer backward.deinit();

    //check n_inputs
    try std.testing.expect(dense_layer.n_inputs == layer1.get_n_inputs());

    //check n_neurons
    try std.testing.expect(dense_layer.n_neurons == layer1.get_n_neurons());

    //utils myDense cast, is the only way to access and anyopaque
    //const myDense: *DenseLayer(f64, allocator) = @ptrCast(@alignCast(layer1.layer_ptr));

    //check get_input
    for (0..dense_layer.input.data.len) |i| {
        try std.testing.expect(dense_layer.input.data[i] == layer1.get_input().data[i]);
    }

    //check get_output
    for (0..dense_layer.output.data.len) |i| {
        try std.testing.expect(dense_layer.output.data[i] == layer1.get_output().data[i]);
    }
}

test "ActivationLayer forward and backward test" {
    std.debug.print("\n     test: ActivationLayer forward and backward test ", .{});
    const allocator = &pkg_allocator.allocator;

    // const argsStruct = struct {
    //     n_inputs: usize,
    //     n_neurons: usize,
    // };

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var activ_layer = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = allocator,
    };
    const layer1 = ActivationLayer(f64).create(&activ_layer);
    // n_input = 5, n_neurons= 4
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 5,
        .n_neurons = 4,
    }));
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, -6.0, -2.0 },
        [_]f64{ -14.0, -15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, -3.0, -1.0 },
        [_]f64{ 4.0, 5.0, -6.0, -2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    const output_tensor = try layer1.forward(&input_tensor);
    for (0..output_tensor.data.len) |i| {
        try std.testing.expect(output_tensor.data[i] >= 0);
    }

    // Test backward
    var res = try layer1.backward(&input_tensor);
    defer res.deinit();
}

test "Conv forward()" {
    std.debug.print("\n     Test: Conv forward test ", .{});

    const allocator = &pkg_allocator.allocator;

    // Define input data
    var input_data = [_]f64{
        // Batch 1, Channel 1
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        // Batch 1, Channel 2
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        // Batch 2, Channel 1
        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        // Batch 2, Channel 2
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    };
    var input_shape = [_]usize{ 2, 2, 3, 3 }; // [batches, in_channels, height, width]
    var input = try Tensor(f64).fromArray(allocator, &input_data, &input_shape);
    defer input.deinit();

    // Create the convolutional layer
    var conv_layer = ConvolutionalLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .kernel_shape = undefined,
        .stride = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    var layer = conv_layer.create();

    // Initialize the convolutional layer
    // input_channels=2, output_channels=2, kernel_shape=[1,1,2,2]
    try layer.init(
        &pkg_allocator.allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 2,
            .kernel_shape = .{ 3, 2, 2, 2 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    defer layer.deinit();

    // Perform the forward pass
    var output = try layer.forward(&input);
    std.debug.print("\nOutput shape: {any}\n", .{output.shape});

    // Print the output for verification
    std.debug.print("\nOutput of forward pass:\n", .{});
    output.info();

    // Create a dummy gradient for the backward pass (same shape as output)
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1; // Set the gradient to 1 for simplicity
    }
    var dValues = try Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    // Perform the backward pass
    // var dInput = try layer.backward(&dValues);
    // defer dInput.deinit();

    // Print the gradients for verification
    std.debug.print("\nWeight gradients:\n", .{});
    // conv_layer.w_gradients.printMultidim();
    conv_layer.w_gradients.info();

    // std.debug.print("Input gradients:\n", .{});
    // dInput.printMultidim();

    // Verify the shapes
    // try std.testing.expectEqual(conv_layer.w_gradients.shape, conv_layer.weights.shape);
    // try std.testing.expectEqual(dInput.shape, input.shape);

    // Clean up resourcess
}

test "Complete test of the Flatten layer functionalities with first dimension unflattened" {
    std.debug.print("\n     Test: Flatten layer forward and backward test ", .{});

    const allocator = &pkg_allocator.allocator;

    var input_data = [_]f64{
        // Batch 1, Channel 1
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        // Batch 1, Channel 2
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        // Batch 2, Channel 1
        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        // Batch 2, Channel 2
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    };
    var input_shape = [_]usize{ 2, 2, 3, 3 }; // [batch_size=2, channels=2, height=3, width=3]
    var input = try Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Create the Flatten layer
    var flatten_layer = FlattenLayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = allocator,
        .original_shape = &[_]usize{},
    };
    var layer = flatten_layer.create();

    // Initialize the Flatten layer with placeholder args
    var init_args = FlattenLayer(f64).FlattenInitArgs{
        .placeholder = true,
    };
    try layer.init(allocator, &init_args);

    // Perform the forward pass
    std.debug.print("\nFlatten forward pass test...\n", .{});
    var output = try layer.forward(&input);
    std.debug.print("Output shape after flatten: {any}\n", .{output.shape});
    std.debug.print("Output of forward pass:\n", .{});
    output.info();

    // Verify the output size (should be [2, 18])
    // Because input is [2,2,3,3], the last dimensions product: 2*3*3=18
    // So output.shape should be [2, 18]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 18), output.shape[1]);

    // Create a dummy gradient for the backward pass (same shape as output: [2, 18])
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1; // Set all gradients to 1 for simplicity
    }
    var dValues = try Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    // Perform the backward pass
    std.debug.print("\nFlatten backward pass test...\n", .{});
    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();
    std.debug.print("dInput shape after backward (should match the original input shape): {any}\n", .{dInput.shape});
    dInput.info();

    // Verify the shapes are the same as original input
    try std.testing.expectEqualSlices(usize, dInput.shape, input.shape);

    // Clean up resources
    layer.deinit();
    _ = &input_data;
    _ = &flatten_layer;
}

test "Pooling layer forward and backward test (Max Pooling 2D, stride=1)" {
    std.debug.print("\n     Test: Pooling layer forward and backward test (Max Pooling, stride=1)\n", .{});

    const allocator = &pkg_allocator.allocator;

    // 1x1x3x3 input (batch x channels x rows x cols)
    var input_data = [_]f64{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        40.0, 50.0, 60.0,
    };
    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input = try tensor.Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Create the Pooling layer (Max Pooling) with kernel=2x2 and stride=1x1
    var pooling_layer = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
        .allocator = allocator,
    };
    var layer = try pooling_layer.create();

    const InitArgs = struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    };

    var init_args = InitArgs{
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
    };

    // Initialize the layer
    try layer.init(allocator, &init_args);
    // Deinitialize the layer to avoid leaks
    defer layer.deinit();

    // Forward pass
    std.debug.print("\nPooling forward pass test...\n", .{});
    var output = try layer.forward(&input);

    std.debug.print("Output shape after pooling: {any}\n", .{output.shape});
    output.info();

    // Check output shape
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]); // batch size
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]); // rows
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]); // cols

    // Check output values
    // Expected:
    // [ [ [ [5,  6],
    //        [50, 60] ] ] ]
    try std.testing.expectEqual(@as(f64, 5.0), output.data[0 * 4 + 0]);
    try std.testing.expectEqual(@as(f64, 6.0), output.data[0 * 4 + 1]);
    try std.testing.expectEqual(@as(f64, 50.0), output.data[0 * 4 + 2]);
    try std.testing.expectEqual(@as(f64, 60.0), output.data[0 * 4 + 3]);

    // Backward pass
    std.debug.print("\nPooling backward pass test...\n", .{});

    // Create dValues (1x1x2x2) and set all gradients to 1
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1.0;
    }
    var dValues = try tensor.Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();

    std.debug.print("dInput shape after backward: {any}\n", .{dInput.shape});
    dInput.info();

    // Check that dInput shape matches the original input
    try std.testing.expectEqualSlices(usize, dInput.shape, input.shape);

    // Expected dInput:
    // [ [ [ [0,0,0],
    //        [0,1,1],
    //        [0,1,1] ] ] ]
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[0]); // (0,0)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[1]); // (0,1)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[2]); // (0,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[3]); // (1,0)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[4]); // (1,1)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[5]); // (1,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[6]); // (2,0)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[7]); // (2,1)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[8]); // (2,2)

}

test "BatchNormLayer forward and backward test" {
    std.debug.print("\n     test: BatchNormLayer forward and backward test", .{});
    const allocator = &pkg_allocator.allocator;

    // Create BatchNormLayer
    var batch_norm = BatchNormLayer(f64){
        .gamma = undefined,
        .beta = undefined,
        .input = undefined,
        .output = undefined,
        .running_mean = undefined,
        .running_var = undefined,
        .epsilon = undefined,
        .momentum = undefined,
        .is_training = undefined,
        .allocator = allocator,
        .gamma_grad = undefined,
        .beta_grad = undefined,
        .normalized = undefined,
        .std_dev = undefined,
        .var_ = undefined,
        .mean = undefined,
    };
    const layer = BatchNormLayer(f64).create(&batch_norm);

    // Initialize with 4 features
    try layer.init(allocator, @constCast(&BatchNormLayer(f64).BatchNormInitArgs{
        .num_features = 4,
        .epsilon = 1e-5,
        .momentum = 0.1,
    }));
    defer layer.deinit();

    // Test input: batch_size=3, num_features=4
    var input_data = [_]f64{
        // Batch 1
        1.0, 2.0, 3.0, 4.0,
        // Batch 2
        2.0, 3.0, 4.0, 5.0,
        // Batch 3
        3.0, 4.0, 5.0, 6.0,
    };
    var input_shape = [_]usize{ 3, 4 };
    var input = try Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Forward pass
    const output = try layer.forward(&input);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 4);

    // Check that output is normalized (mean close to 0, variance close to 1)

    // Test backward pass
    var grad_data = [_]f64{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
    };
    var grad = try Tensor(f64).fromArray(allocator, &grad_data, input_shape[0..]);
    defer grad.deinit();

    var dx = try layer.backward(&grad);
    defer dx.deinit();

    // Check gradient shapes
    try std.testing.expectEqual(dx.shape[0], input.shape[0]);
    try std.testing.expectEqual(dx.shape[1], input.shape[1]);

    // Check that gamma_grad and beta_grad are updated
    const bn_layer: *BatchNormLayer(f64) = @ptrCast(@alignCast(layer.layer_ptr));
    try std.testing.expect(bn_layer.gamma_grad.data.len > 0);
    try std.testing.expect(bn_layer.beta_grad.data.len > 0);
}
