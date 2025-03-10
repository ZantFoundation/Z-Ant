const std = @import("std");
const zant = @import("zant");
const tensor = zant.core.tensor;
const Layer = zant.model.layer;
const DenseLayer = Layer.DenseLayer;
const Model = zant.model.Model;
const ActivationType = Layer.ActivationType;
const Trainer = zant.data_handler.trainer;
const pkgAllocator = zant.utils.allocator;

test "Model with multiple Denselayers forward test" {
    std.debug.print("\n     test: Model with multiple layers forward test", .{});
    const allocator = pkgAllocator.allocator;

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    defer model.deinit();

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
    try model.addLayer(layer1_);

    var dense_layer2 = DenseLayer(f64){
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
    var layer2_ = DenseLayer(f64).create(&dense_layer2);
    try layer2_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 3,
        }),
    );
    try model.addLayer(layer2_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    _ = try model.forward(&input_tensor);
}

test {
    _ = @import("test_lossFunction.zig");
    _ = @import("test_activation_function.zig");
    _ = @import("test_lossFunction.zig");
    _ = @import("Layers/test_layers.zig");
}
