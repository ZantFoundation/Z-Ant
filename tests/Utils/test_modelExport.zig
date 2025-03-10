const std = @import("std");
const zant = @import("zant");
const model_import_export = zant.utils.model;
const Model = zant.model.Model;
const Layer = zant.model.layer;
const DenseLayer = Layer.DenseLayer;
const ActivationLayer = Layer.ActivationLayer;
const ConvolutionalLayer = Layer.ConvolutionalLayer;
const FlattenLayer = Layer.FlattenLayer;
const PoolingLayer = Layer.poolingLayer.PoolingLayer;

const Tensor = zant.core.tensor.Tensor;
const ActivationType = Layer.ActivationType;
const Trainer = zant.data_handler.trainer;
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;

test "Import/Export of a tensor" {
    std.debug.print("\n     test: Import/Export of a tensor", .{});

    const allocator = pkgAllocator.allocator; //std.heap.page_allocator; // Known memory leak, avoid testing allocator
    const file_path = "importExportTensorTestFile.bin";
    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try model_import_export.exportTensor(f32, t1, writer);
    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();
    var t2: Tensor(f32) = try model_import_export.importTensor(f32, &allocator, reader);
    defer t2.deinit();

    file.close();

    //same data
    try std.testing.expect(t1.data.len == t2.data.len);
    for (0..t1.data.len) |i| {
        try std.testing.expect(t1.data[i] == t2.data[i]);
    }

    //same size
    try std.testing.expect(t1.size == t2.size);

    //same shape
    try std.testing.expect(t1.shape.len == t2.shape.len);
    for (0..t1.shape.len) |i| {
        try std.testing.expect(t1.shape[i] == t2.shape[i]);
    }

    try std.fs.cwd().deleteFile(file_path);
}

test "Import/Export of dense layer" {
    std.debug.print("\n     test: Import/Export of dense layer", .{});
    const allocator = std.heap.raw_c_allocator; //OSS!! denseLayerPtr in importLayer() is not freed
    const file_path = "importExportDenseLayerTestFile.bin";

    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var dense_layer1 = DenseLayer(f64){
        .weights = undefined,
        .input = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer1_ = DenseLayer(f64).create(&dense_layer1);
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 3,
            .n_neurons = 2,
        }),
    );
    defer layer1_.deinit();

    try model_import_export.exportLayer(f64, layer1_, writer);

    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();

    var layer_imported = try model_import_export.importLayer(f64, &allocator, reader);
    defer layer_imported.deinit();

    file.close();

    //same type
    std.debug.print("\n same type: {any}={any}", .{ layer_imported.layer_type, layer1_.layer_type });
    try std.testing.expect(layer_imported.layer_type == layer1_.layer_type);

    //same n_neurons
    std.debug.print("\n same n_neurons: {any}={any}", .{ layer_imported.get_n_neurons(), layer1_.get_n_neurons() });
    try std.testing.expect(layer_imported.get_n_neurons() == layer1_.get_n_neurons());

    //same n_inputs
    std.debug.print("\n same n_inputs  {any}={any}", .{ layer_imported.get_n_inputs(), layer1_.get_n_inputs() });
    try std.testing.expect(layer_imported.get_n_inputs() == layer1_.get_n_inputs());

    //check layer
    const denseImportedPtr: *DenseLayer(f64) = @alignCast(@ptrCast(layer_imported.layer_ptr));
    //same weights len
    try std.testing.expect(denseImportedPtr.weights.data.len == dense_layer1.weights.data.len);
    //same weights
    for (0..denseImportedPtr.weights.data.len) |i| {
        try std.testing.expect(denseImportedPtr.weights.data[i] == dense_layer1.weights.data[i]);
    }
    //same bias len
    try std.testing.expect(denseImportedPtr.bias.data.len == dense_layer1.bias.data.len);
    //same bias
    for (0..denseImportedPtr.bias.data.len) |i| {
        try std.testing.expect(denseImportedPtr.bias.data[i] == dense_layer1.bias.data[i]);
    }

    try std.fs.cwd().deleteFile(file_path);
}

test "Import/Export of convolutional layer" {
    std.debug.print("\n     test: Import/Export of convolutional layer", .{});
    const allocator = std.heap.raw_c_allocator; //OSS!! denseLayerPtr in importLayer() is not freed
    const file_path = "importExportDenseLayerTestFile.bin";

    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var conv_layer0 = ConvolutionalLayer(f64){
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
    var layer0_ = conv_layer0.create();
    try layer0_.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 16,
            .kernel_shape = .{ 16, 16, 3, 3 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    defer layer0_.deinit();

    try model_import_export.exportLayer(f64, layer0_, writer);

    file.close();

    // IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();

    var layer_imported = try model_import_export.importLayer(f64, &allocator, reader);
    defer layer_imported.deinit();

    file.close();

    // same type
    std.debug.print("\n same type: {any}={any}", .{ layer_imported.layer_type, layer0_.layer_type });
    try std.testing.expect(layer_imported.layer_type == layer0_.layer_type);

    // check layer
    const convImportedPtr: *ConvolutionalLayer(f64) = @alignCast(@ptrCast(layer_imported.layer_ptr));

    // same weights
    try std.testing.expect(TensMath.equal(f64, &conv_layer0.weights, &convImportedPtr.weights) == true);

    // same bias
    try std.testing.expect(TensMath.equal(f64, &conv_layer0.bias, &convImportedPtr.bias) == true);

    // same input_channels
    try std.testing.expectEqual(conv_layer0.input_channels, convImportedPtr.input_channels);

    // same kernel_shape
    try std.testing.expectEqual(conv_layer0.kernel_shape[0], convImportedPtr.kernel_shape[0]);
    try std.testing.expectEqual(conv_layer0.kernel_shape[1], convImportedPtr.kernel_shape[1]);
    try std.testing.expectEqual(conv_layer0.kernel_shape[2], convImportedPtr.kernel_shape[2]);
    try std.testing.expectEqual(conv_layer0.kernel_shape[3], convImportedPtr.kernel_shape[3]);

    // same stride
    try std.testing.expectEqual(conv_layer0.stride[0], convImportedPtr.stride[0]);
    try std.testing.expectEqual(conv_layer0.stride[1], convImportedPtr.stride[1]);

    try std.fs.cwd().deleteFile(file_path);
}

test "Import/Export of activation layer" {
    std.debug.print("\n     test: Import/Export of activation layer", .{});
    const allocator = std.heap.raw_c_allocator;
    const file_path = "importExportTestFile.bin";
    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var activ_layer = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    const layer1_ = ActivationLayer(f64).create(&activ_layer);
    // n_input = 5, n_neurons= 4
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 5,
            .n_neurons = 4,
        }),
    );

    //defer layer1_.deinit();

    try model_import_export.exportLayer(f64, layer1_, writer);

    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();

    var layer_imported = try model_import_export.importLayer(f64, &allocator, reader);
    //defer layer_imported.deinit();

    file.close();

    //same type layer
    std.debug.print("\n same type: {any}={any}", .{ layer_imported.layer_type, layer1_.layer_type });
    try std.testing.expect(layer_imported.layer_type == layer1_.layer_type);

    const actImportedPtr: *ActivationLayer(f64) = @alignCast(@ptrCast(layer_imported.layer_ptr));
    //same type activation
    std.debug.print("\n same type: {any}={any}", .{ actImportedPtr.activationFunction, activ_layer.activationFunction });
    try std.testing.expect(actImportedPtr.activationFunction == activ_layer.activationFunction);

    //same n_neurons
    std.debug.print("\n same n_neurons: {any}={any}", .{ layer_imported.get_n_neurons(), layer1_.get_n_neurons() });
    try std.testing.expect(layer_imported.get_n_neurons() == layer1_.get_n_neurons());

    //same n_inputs
    std.debug.print("\n same n_inputs  {any}={any}", .{ layer_imported.get_n_inputs(), layer1_.get_n_inputs() });
    try std.testing.expect(layer_imported.get_n_inputs() == layer1_.get_n_inputs());

    try std.fs.cwd().deleteFile(file_path);
}

test "Export of a DENSE model" {
    std.debug.print("\n     test: Export of a DENSE model", .{});

    const allocator = std.heap.raw_c_allocator; //
    const file_path = "importExportTestFile.bin";

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    defer model.deinit();

    //layer 0: 3 inputs, 2 neurons
    var layer0 = DenseLayer(f64){
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
    var layer0_ = DenseLayer(f64).create(&layer0);
    try layer0_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 3,
            .n_neurons = 2,
        }),
    );
    try model.addLayer(layer0_);

    //layer 1: 2 inputs, 5 neurons
    var layer1 = DenseLayer(f64){
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
    var layer1_ = DenseLayer(f64).create(&layer1);
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 5,
        }),
    );
    try model.addLayer(layer1_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var targetArray: [2][5]f64 = [_][5]f64{
        [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 },
        [_]f64{ 4.0, 5.0, 6.0, 4.0, 5.0 },
    };
    var targetShape: [2]usize = [_]usize{ 2, 5 };

    var input_tensor = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    var target_tensor = try Tensor(f64).fromArray(&allocator, &targetArray, &targetShape);
    defer {
        target_tensor.deinit();
        std.debug.print("\n -.-.-> target_tensor deinitialized", .{});
    }

    try Trainer.trainTensors(
        f64, //type
        &allocator, //allocator
        &model, //model
        &input_tensor, //input
        &target_tensor, //target
        1, //epochs
        0.5, //learning rate
    );

    try model_import_export.exportModel(f64, model, file_path);

    var imported_model = try model_import_export.importModel(f64, &allocator, file_path);
    defer imported_model.deinit();

    //check same layers
    std.debug.print("\n n layer model:{}, n layer imported_model:{}", .{ model.layers.items.len, imported_model.layers.items.len });
    try std.testing.expect(imported_model.layers.items.len == model.layers.items.len);
    for (0..imported_model.layers.items.len) |i| {
        try std.testing.expect(imported_model.layers.items[i].layer_type == model.layers.items[i].layer_type);
    }

    //same layer 0
    const l0_export: *DenseLayer(f64) = @ptrCast(@alignCast(model.layers.items[0].layer_ptr));
    const l0_import: *DenseLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[0].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l0_export.n_inputs == l0_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l0_export.n_neurons == l0_import.n_neurons);

    //  same weights
    try std.testing.expect(TensMath.equal(f64, &l0_export.weights, &l0_import.weights) == true);

    //  same bias
    try std.testing.expect(TensMath.equal(f64, &l0_export.bias, &l0_import.bias) == true);

    //same layer 1
    const l1_export: *DenseLayer(f64) = @ptrCast(@alignCast(model.layers.items[1].layer_ptr));
    const l1_import: *DenseLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[1].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l1_export.n_inputs == l1_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l1_export.n_neurons == l1_import.n_neurons);

    //  same weights
    try std.testing.expect(TensMath.equal(f64, &l1_export.weights, &l1_import.weights) == true);

    //  same bias
    try std.testing.expect(TensMath.equal(f64, &l1_export.bias, &l1_import.bias) == true);

    try Trainer.trainTensors(
        f64, //type
        &allocator, //allocator
        &imported_model, //model
        &input_tensor, //input
        &target_tensor, //target
        1, //epochs
        0.5, //learning rate
    );

    try std.fs.cwd().deleteFile(file_path);
}

test "Export/Import of a DENSE+ACTIVATION model" {
    std.debug.print("\n     test: Export of a DENSE+ACTIVATION model", .{});

    const allocator = std.heap.raw_c_allocator;
    const file_path = "importExportTestFile.bin";

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    // defer {
    //     std.debug.print("\n ------------------------------- model.deinit(); ", .{});
    //     model.deinit();
    // }

    //layer 0: 3 inputs, 2 neurons
    var layer0 = DenseLayer(f64){
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
    var layer0_ = DenseLayer(f64).create(&layer0);
    try layer0_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 3,
            .n_neurons = 2,
        }),
    );
    try model.addLayer(layer0_);

    //layer 1: 3 inputs, 2 neurons
    var layer1 = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer1_ = ActivationLayer(f64).create(&layer1);
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 2,
        }),
    );
    try model.addLayer(layer1_);

    //layer 2: 2 inputs, 5 neurons
    var layer2 = DenseLayer(f64){
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
    var layer2_ = DenseLayer(f64).create(&layer2);
    try layer2_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 5,
        }),
    );
    try model.addLayer(layer2_);

    var layer3 = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer3_ = ActivationLayer(f64).create(&layer3);
    try layer3_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 5,
        }),
    );
    try model.addLayer(layer3_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var targetArray: [2][5]f64 = [_][5]f64{
        [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 },
        [_]f64{ 4.0, 5.0, 6.0, 4.0, 5.0 },
    };
    var targetShape: [2]usize = [_]usize{ 2, 5 };

    var input_tensor = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    var target_tensor = try Tensor(f64).fromArray(&allocator, &targetArray, &targetShape);
    defer {
        target_tensor.deinit();
        std.debug.print("\n -.-.-> target_tensor deinitialized", .{});
    }

    try Trainer.trainTensors(
        f64, //type
        &allocator, //allocator
        &model, //model
        &input_tensor, //input
        &target_tensor, //target
        1, //epochs
        0.5, //learning rate
    );

    try model_import_export.exportModel(f64, model, file_path);

    var imported_model = try model_import_export.importModel(f64, &allocator, file_path);
    // defer {
    //     std.debug.print("\n ------------------------------- imported_model.deinit(); ", .{});
    //     imported_model.deinit();
    // }

    //check same layers
    std.debug.print("\n n layer model:{}, n layer imported_model:{}", .{ model.layers.items.len, imported_model.layers.items.len });
    try std.testing.expect(imported_model.layers.items.len == model.layers.items.len);
    for (0..imported_model.layers.items.len) |i| {
        try std.testing.expect(imported_model.layers.items[i].layer_type == model.layers.items[i].layer_type);
        std.debug.print("\n\ntype:{}", .{imported_model.layers.items[i].layer_type});
    }

    // --------same layer 0
    const l0_export: *DenseLayer(f64) = @ptrCast(@alignCast(model.layers.items[0].layer_ptr));
    const l0_import: *DenseLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[0].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l0_export.n_inputs == l0_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l0_export.n_neurons == l0_import.n_neurons);

    //  same weights
    try std.testing.expect(TensMath.equal(f64, &l0_export.weights, &l0_import.weights) == true);

    //  same bias
    try std.testing.expect(TensMath.equal(f64, &l0_export.bias, &l0_import.bias) == true);

    // --------same layer 1
    const l1_export: *ActivationLayer(f64) = @ptrCast(@alignCast(model.layers.items[1].layer_ptr));
    const l1_import: *ActivationLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[1].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l1_export.n_inputs == l1_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l1_export.n_neurons == l1_import.n_neurons);

    //  same activationFunction
    try std.testing.expect(l1_export.activationFunction == l1_import.activationFunction);

    // --------same layer 2
    const l2_export: *DenseLayer(f64) = @ptrCast(@alignCast(model.layers.items[2].layer_ptr));
    const l2_import: *DenseLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[2].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l2_export.n_inputs == l2_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l2_export.n_neurons == l2_import.n_neurons);

    //  same weights
    try std.testing.expect(TensMath.equal(f64, &l2_export.weights, &l2_import.weights) == true);

    //  same bias
    try std.testing.expect(TensMath.equal(f64, &l2_export.bias, &l2_import.bias) == true);

    // --------same layer 3
    const l3_export: *ActivationLayer(f64) = @ptrCast(@alignCast(model.layers.items[3].layer_ptr));
    const l3_import: *ActivationLayer(f64) = @ptrCast(@alignCast(imported_model.layers.items[3].layer_ptr));

    //  same n_inputs
    try std.testing.expect(l3_export.n_inputs == l3_import.n_inputs);

    //  same n_neurons
    try std.testing.expect(l3_export.n_neurons == l3_import.n_neurons);

    //  same activationFunction
    try std.testing.expect(l3_export.activationFunction == l3_import.activationFunction);

    try std.fs.cwd().deleteFile(file_path);

    std.debug.print("\n ------------------------------- model.deinit(); ", .{});
    model.deinit();

    std.debug.print("\n ------------------------------- imported_model.deinit(); ", .{});
    imported_model.deinit();
}

test "Export/Import of a COMPLEX model" {
    std.debug.print("\n     test: Export/Import of a COMPLEX model", .{});

    const allocator = std.heap.raw_c_allocator;
    const file_path = "importExportTestFile.bin";

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    defer {
        std.debug.print("\n ------------------------------- imported_model.deinit(); ", .{});
        model.deinit();
    }

    //layer 0 ----------------------------------------------------------------------------
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
        .allocator = &allocator,
    };
    var layer_ = conv_layer.create();
    try layer_.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 1,
            .kernel_shape = .{ 16, 1, 2, 2 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer_);

    //layer 1 ----------------------------------------------------------------------------
    var conv_layer2 = ConvolutionalLayer(f64){
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
    var layer2_ = conv_layer2.create();
    try layer2_.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 16,
            .kernel_shape = .{ 16, 16, 3, 3 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer2_);

    //layer 2 ----------------------------------------------------------------------------
    var flatten_layer = FlattenLayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = &allocator,
        .original_shape = &[_]usize{},
    };
    var Flattenlayer = flatten_layer.create();

    // Initialize the Flatten layer with placeholder args
    var init_argsF = FlattenLayer(f64).FlattenInitArgs{
        .placeholder = true,
    };
    try Flattenlayer.init(&allocator, &init_argsF);

    try model.addLayer(Flattenlayer);

    //layer 3 ----------------------------------------------------------------------------
    var layer3 = DenseLayer(f64){
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
    var layer3_ = DenseLayer(f64).create(&layer3);
    try layer3_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10000,
        .n_neurons = 256,
    }));
    try model.addLayer(layer3_);

    //layer 4 ----------------------------------------------------------------------------
    var layer3Activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer3_act = ActivationLayer(f64).create(&layer3Activ);
    try layer3_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10000,
        .n_neurons = 256,
    }));
    try model.addLayer(layer3_act);

    //new dense layer

    //layer 5 ----------------------------------------------------------------------------
    var layer4 = DenseLayer(f64){
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

    var layer4_ = DenseLayer(f64).create(&layer4);
    try layer4_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));

    try model.addLayer(layer4_);

    //layer 6 ----------------------------------------------------------------------------
    var layer4Activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer4_act = ActivationLayer(f64).create(&layer4Activ);
    try layer4_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));
    try model.addLayer(layer4_act);

    try model_import_export.exportModel(f64, model, file_path);

    var imported_model = try model_import_export.importModel(f64, &allocator, file_path);
    defer {
        std.debug.print("\n ------------------------------- imported_model.deinit(); ", .{});
        imported_model.deinit();
    }

    //check same layers
    std.debug.print("\n n layer model:{}, n layer imported_model:{}", .{ model.layers.items.len, imported_model.layers.items.len });
    try std.testing.expect(imported_model.layers.items.len == model.layers.items.len);
    for (0..imported_model.layers.items.len) |i| {
        try std.testing.expect(imported_model.layers.items[i].layer_type == model.layers.items[i].layer_type);
        std.debug.print("\n\ntype:{}", .{imported_model.layers.items[i].layer_type});
    }

    try std.fs.cwd().deleteFile(file_path);
}
