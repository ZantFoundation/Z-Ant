const std = @import("std");
const zant = @import("../zant.zig");
const tensor = zant.core.tensor;
const layer = zant.model.layer;
const Model = zant.model;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const DenseLayer = layer.DenseLayer;
const ConvolutionalLayer = layer.ConvolutionalLayer; // Make sure this is the correct import for your convolutional layer

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

pub const Optimizers = enum {
    SGD,
    Adam,
    RMSprop,
};

/// Generic Optimizer factory
pub fn Optimizer(comptime T: type, comptime Xtype: type, comptime YType: type, func: fn (comptime type, comptime type, comptime type, f64) type, lr: f64) type {
    const optim = func(T, Xtype, YType, lr){};
    return struct {
        optimizer: func(T, Xtype, YType, lr) = optim,

        /// Call the optimizer's step function
        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            try self.optimizer.step(model);
        }
    };
}

/// SGD Optimizer
/// If new layers are added, update this logic accordingly
pub fn optimizer_SGD(T: type, XType: type, YType: type, lr: f64) type {
    _ = XType;
    _ = YType;

    return struct {
        learning_rate: f64 = lr,

        /// Step through the model, updating parameters
        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            var counter: u32 = 0;
            for (model.layers.items) |layer_| {
                switch (layer_.layer_type) {
                    .DenseLayer => {
                        const myDense: *DenseLayer(T) = @ptrCast(@alignCast(layer_.layer_ptr));
                        const weight_gradients = &myDense.w_gradients;
                        const bias_gradients = &myDense.b_gradients;
                        const weight = &myDense.weights;
                        const bias = &myDense.bias;

                        std.debug.print("\n------ step {} (DenseLayer)", .{counter});
                        //weight_gradients.info();
                        try self.update_tensor(weight, weight_gradients);
                        try self.update_tensor(bias, bias_gradients);
                    },
                    .ConvolutionalLayer => {
                        const myConv: *ConvolutionalLayer(T) = @ptrCast(@alignCast(layer_.layer_ptr));
                        const kernel_gradients = &myConv.w_gradients;
                        const bias_gradients = &myConv.b_gradients;
                        const kernel = &myConv.weights;
                        const bias = &myConv.bias;

                        std.debug.print("\n------ step {} (ConvLayer)", .{counter});
                        // kernel_gradients.info();
                        // kernel.info();
                        try self.update_tensor(kernel, kernel_gradients);
                        try self.update_tensor(bias, bias_gradients);
                    },
                    else => {
                        //print layer name
                        std.debug.print("\n------ step {} (LayerType not supported)", .{counter});
                    },
                }
                counter += 1;
            }
        }

        /// Helper function to update any parameter tensor
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;

            // Gradient descent step
            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

/// Example of another optimizer (e.g. Adam) could be implemented similarly
pub fn optimizer_ADAMTEST(T: type, lr: f64) type {
    return struct {
        learning_rate: f64 = lr,

        // Minimal example (not fully implemented)
        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            for (model.layers) |*some_layer| {
                // Assuming a certain layer structure with w_gradients
                // This is just an example; adapt as needed.
                if (some_layer.layer_type == layer.LayerType.DenseLayer) {
                    const dense: *DenseLayer(T) = @ptrCast(@alignCast(some_layer.layer_ptr));
                    const weight_gradients = &dense.w_gradients;
                    try self.update_tensor(&dense.weights, weight_gradients);
                } else if (some_layer.layer_type == layer.LayerType.ConvolutionalLayer) {
                    const conv: *ConvolutionalLayer(T) = @ptrCast(@alignCast(some_layer.layer_ptr));
                    const kernel_gradients = &conv.w_gradients;
                    try self.update_tensor(&conv.weights, kernel_gradients);
                }
            }
        }

        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}
