pub const optim = @import("optim.zig");
pub const loss_function = @import("lossFunction.zig");
pub const layer = @import("Layers/layer.zig");

const std = @import("std");
const zant = @import("../zant.zig");
const tensor = zant.core.tensor;
const TensMath = tensor.math_standard;
const Optim = @import("optim.zig");
const DataProc = zant.data_handler.data_processor;
const NormalizType = DataProc.NormalizationType;

/// The `Model` struct represents a neural network model composed of multiple layers.
/// This model can be configured with a specific data type (`T`) and allocator. It supports
/// adding layers, running forward and backward passes, and manages the allocation and
/// deallocation of resources.
pub fn Model(comptime T: type) type {
    return struct {
        layers: std.ArrayList(layer.Layer(T)) = undefined, // Array of layers in the model.
        allocator: *const std.mem.Allocator, // Allocator reference for dynamic memory allocation.
        input_tensor: tensor.Tensor(T), // Tensor that holds the model's input data.

        /// Initializes the model, setting up an empty list of layers and initializing
        /// the input tensor.
        ///
        /// # Errors
        /// Returns an error if memory allocation for the `layers` array or `input_tensor` fails.
        pub fn init(self: *@This()) !void {
            self.layers = std.ArrayList(layer.Layer(T)).init(self.allocator.*);
            self.input_tensor = try tensor.Tensor(T).init(self.allocator);
        }

        /// Deinitializes the model, releasing memory for each layer and the input tensor.
        ///
        /// This method iterates through each layer, deinitializes it, and then frees
        /// the layer array and input tensor memory.
        pub fn deinit(self: *@This()) void {
            for (self.layers.items, 0..) |*layer_, i| {
                std.debug.print("\n deinitializing layer {} ... ", .{i});
                layer_.deinit();
                std.debug.print("->  layer {} deinitialized", .{i});
            }
            self.layers.deinit();
            std.debug.print("\n -.-.-> model layers deinitialized", .{});

            self.input_tensor.deinit(); // pay attention! input_tensor is initialised only if forward() is run at least once.
            std.debug.print("\n -.-.-> model input_tensor deinitialized", .{});
        }

        /// Adds a new layer to the model.
        ///
        /// # Parameters
        /// - `new_layer`: A pointer to the new layer to add to the model.
        ///
        /// # Errors
        /// Returns an error if reallocating the `layers` array fails.
        pub fn addLayer(self: *@This(), new_layer: layer.Layer(T)) !void {
            try self.layers.append(new_layer);
        }

        /// Executes the forward pass through the model with the specified input tensor.
        ///
        /// # Parameters
        /// - `input`: A pointer to the input tensor.
        ///
        /// # Returns
        /// A pointer to the output tensor of the model after the forward pass.
        ///
        /// # Errors
        /// Returns an error if any layer's forward pass or tensor copying fails.
        pub fn forward(self: *@This(), input_tensor: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            if (self.layers.items.len == 0) {
                if (self.input_tensor.data.len > 0) self.input_tensor.deinit();

                self.input_tensor = try input_tensor.copy();
                return &self.input_tensor;
            }

            // Store input tensor

            if (self.input_tensor.data.len > 0) self.input_tensor.deinit();

            self.input_tensor = try input_tensor.copy();

            // Forward pass through all layers
            var i: usize = 0;
            while (i < self.layers.items.len) : (i += 1) {
                const current_layer = self.layers.items[i];
                std.debug.print("--------------------------------------forwarding layer {d}\n", .{i});
                if (current_layer.layer_type != layer.LayerType.ActivationLayer) try DataProc.normalize(T, self.getPrevOut(i), NormalizType.UnityBasedNormalizartion);
                // DEBUG METHOD try self.getPrevOut(i).isSafe();
                _ = try current_layer.forward(self.getPrevOut(i));
                //print the output shape
                std.debug.print(" output shape: {any}\n", .{current_layer.get_output().shape});
                //self.layers.items[i].printLayer(0);
            }

            // Return a copy of the final output

            if (self.input_tensor.data.len > 0) self.input_tensor.deinit();

            self.input_tensor = try (self.getPrevOut(self.layers.items.len)).copy();
            return &self.input_tensor;
        }

        /// Executes the backward pass through the model with the specified gradient tensor.
        ///
        /// # Parameters
        /// - `gradient`: A pointer to the gradient tensor to backpropagate.
        ///
        /// # Returns
        /// A pointer to the final gradient tensor after the backward pass.
        ///
        /// # Errors
        /// Returns an error if any layer's backward pass or tensor copying fails.
        pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !void {
            var grad_ptr: tensor.Tensor(T) = undefined;
            defer grad_ptr.deinit();
            var grad_duplicate: tensor.Tensor(T) = try gradient.copy();
            defer grad_duplicate.deinit();

            var counter = self.layers.items.len - 1;
            while (counter >= 0) : (counter -= 1) {
                std.debug.print("\n--------------------------------------backwarding layer {}", .{counter});
                if (counter != self.layers.items.len - 1) grad_ptr.deinit();
                grad_ptr = try self.layers.items[counter].backward(&grad_duplicate);

                // Conserviamo la vecchia copia per poi deallocarla
                var old_dup = grad_duplicate;

                // Creiamo la nuova copia da grad_ptr
                grad_duplicate = try grad_ptr.copy();

                // Ora possiamo deallocare la vecchia copia
                old_dup.deinit();

                if (counter == 0) break;
            }

            // Alla fine del ciclo, grad_duplicate contiene l'ultima copia. Non serve più.
            grad_duplicate.deinit();
        }

        /// Retrieves the output of the specified layer or the input tensor for the first layer.
        ///
        /// # Parameters
        /// - `layer_numb`: The index of the layer whose output tensor is to be retrieved.
        ///
        /// # Returns
        /// A pointer to the output tensor of the specified layer, or the input tensor if
        /// `layer_numb` is zero.
        ///
        /// # Errors
        /// Returns an error if the index is out of bounds or other tensor-related errors occur.
        fn getPrevOut(self: *@This(), layer_numb: usize) *tensor.Tensor(T) {
            if (layer_numb == 0) {
                return &self.input_tensor;
            } else {
                return self.layers.items[layer_numb - 1].get_output(); //self.layers[layer_numb - 1].get_outputActivation(); //
            }
        }
    };
}