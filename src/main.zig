const std = @import("std");

const zant = @import("zant");

const core_mod = zant.core;
const tensor = core_mod.tensor;
//--- layers
const model_mod = zant.model;
const layer = model_mod.layer;
const DenseLayer = layer.DenseLayer;
const BatchNormLayer = layer.BatchNormLayer;
const ConvolutionalLayer = layer.ConvolutionalLayer;
const FlattenLayer = layer.FlattenLayer;
const PoolingLayer = layer.PoolingLayer;
const PoolingType = layer.poolingLayer.PoolingType;
const ActivationLayer = layer.ActivationLayer;
const ActivationType = layer.ActivationType;
//--- other
const Model = model_mod.Model;

const data_handler_mod = zant.data_handler;
const loader = data_handler_mod.data_loader;

const Trainer = data_handler_mod.trainer;
const LossType = model_mod.loss_function.LossType;

const onnx = zant.onnx;

const utils_mod = zant.utils;

const allocator = utils_mod.allocator.allocator;

pub fn main() !void {
    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    // First Convolutional Layer (32 filters)
    var conv1 = ConvolutionalLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .kernel_shape = undefined,
        .stride = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = &allocator,
    };
    var conv1_layer = conv1.create();
    try conv1_layer.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 1,
            .kernel_shape = .{ 32, 1, 3, 3 }, // 32 filters, 1 channel, 3x3 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(conv1_layer);

    // ReLU after first conv
    var conv1_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 32 * 26 * 26,
        .n_neurons = 32 * 26 * 26,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var conv1_act = ActivationLayer(f64).create(&conv1_activ);
    try conv1_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 32 * 26 * 26,
        .n_neurons = 32 * 26 * 26,
    }));
    try model.addLayer(conv1_act);

    // First MaxPool
    var pool1 = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
        .allocator = &allocator,
    };
    var pool1_layer = try pool1.create();
    try pool1_layer.init(&allocator, @constCast(&struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    }{
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
    }));
    try model.addLayer(pool1_layer);

    // Second Convolutional Layer (64 filters)
    var conv2 = ConvolutionalLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .kernel_shape = undefined,
        .stride = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = &allocator,
    };
    var conv2_layer = conv2.create();
    try conv2_layer.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 32,
            .kernel_shape = .{ 64, 32, 3, 3 }, // 64 filters, 32 channels, 3x3 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(conv2_layer);

    // ReLU after second conv
    var conv2_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 64 * 11 * 11,
        .n_neurons = 64 * 11 * 11,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var conv2_act = ActivationLayer(f64).create(&conv2_activ);
    try conv2_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 64 * 11 * 11,
        .n_neurons = 64 * 11 * 11,
    }));
    try model.addLayer(conv2_act);

    // Second MaxPool
    var pool2 = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
        .allocator = &allocator,
    };
    var pool2_layer = try pool2.create();
    try pool2_layer.init(&allocator, @constCast(&struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    }{
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
    }));
    try model.addLayer(pool2_layer);

    // Flatten Layer
    var flatten = FlattenLayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = &allocator,
        .original_shape = &[_]usize{},
    };
    var flatten_layer = flatten.create();
    try flatten_layer.init(&allocator, @constCast(&struct { placeholder: bool }{
        .placeholder = true,
    }));
    try model.addLayer(flatten_layer);

    // Dense Layer (512 neurons)
    var dense1 = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var dense1_layer = DenseLayer(f64).create(&dense1);
    try dense1_layer.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 64 * 5 * 5, // After two maxpools: 28->13->5, with 64 channels
        .n_neurons = 512,
    }));
    try model.addLayer(dense1_layer);

    // ReLU after dense
    var dense1_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 512,
        .n_neurons = 512,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var dense1_act = ActivationLayer(f64).create(&dense1_activ);
    try dense1_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 512,
        .n_neurons = 512,
    }));
    try model.addLayer(dense1_act);

    // Output Layer (10 neurons)
    var output = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var output_layer = DenseLayer(f64).create(&output);
    try output_layer.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 512,
        .n_neurons = 10,
    }));
    try model.addLayer(output_layer);

    // Softmax activation
    var softmax = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 10,
        .n_neurons = 10,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var softmax_layer = ActivationLayer(f64).create(&softmax);
    try softmax_layer.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10,
        .n_neurons = 10,
    }));
    try model.addLayer(softmax_layer);

    // Setup DataLoader
    var load = loader.DataLoader(f64, u8, u8, 64, 3){ // Increased batch size to 128
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    try load.loadMNIST2DDataParallel(&allocator, image_file_name, label_file_name);

    // Train with optimized hyperparameters
    try Trainer.TrainDataLoader2D(
        f64,
        u8,
        u8,
        &allocator,
        64, // Increased batch size
        784,
        &model,
        &load,
        30, // Reduced epochs since the model converges faster
        LossType.CCE,
        0.005, // Reduced learning rate for better stability
        0.9, // Increased momentum
        0.0001, // L2 regularization
        1.0,
    );

    model.deinit();
}
