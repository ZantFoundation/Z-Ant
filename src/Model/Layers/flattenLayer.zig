const std = @import("std");

const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor;
const TensMath = zant.core.tensor.math_standard;
const Layer = zant.model.layer;
const LayerError = zant.utils.error_handler.LayerError;

/// Represents a flattening layer in a neural network.
/// This layer reshapes a multidimensional input tensor into a one-dimensional output tensor,
/// often used to transition from convolutional layers to fully connected layers.
///
/// @param T The data type of the tensor elements (e.g., `f32`, `f64`, etc.).
pub fn FlattenLayer(comptime T: type) type {
    return struct {
        // Flatten layer parameters
        input: Tensor.Tensor(T), // Kept for compatibility with tests
        output: Tensor.Tensor(T), // Flattened output
        original_shape: []usize, // Store original shape for backward pass
        allocator: *const std.mem.Allocator,

        const Self = @This();

        // Placeholder struct for init arguments
        pub const FlattenInitArgs = struct {
            placeholder: bool,
        };

        pub fn create(self: *Self) Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.FlattenLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .deinit = deinit,
                    .forward = forward,
                    .backward = backward,
                    .printLayer = printLayer,
                    .get_n_inputs = get_n_inputs,
                    .get_n_neurons = get_n_neurons,
                    .get_input = get_input,
                    .get_output = get_output,
                },
            };
        }

        /// Initialize the Flatten layer (just store allocator, no weights)
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const FlattenInitArgs = @ptrCast(@alignCast(args));
            _ = argsStruct; // We don't really need the placeholder

            self.allocator = alloc;
            self.original_shape = &[_]usize{};
            self.input = try Tensor.Tensor(T).init(alloc);
            self.output = try Tensor.Tensor(T).init(alloc);
            return;
        }

        /// Deallocate the Flatten layer resources
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.original_shape.len > 0) {
                self.allocator.free(self.original_shape);
                self.original_shape = &[_]usize{};
            }
        }

        /// Forward pass: Flatten all dimensions except the first (batch) dimension
        /// Input: [N, D1, D2, ..., Dk]
        /// Output: [N, D1*D2*...*Dk]
        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (input.shape.len < 2) {
                return LayerError.InvalidParameters;
            }

            // Store original shape for backward pass
            if (self.original_shape.len > 0) {
                self.allocator.free(self.original_shape);
            }
            self.original_shape = try self.allocator.dupe(usize, input.shape);
            errdefer self.allocator.free(self.original_shape);

            const batch_size = input.shape[0];
            var total_size: usize = 1;
            for (input.shape[1..]) |dim| {
                total_size *= dim;
            }

            // New shape: [N, total_size]
            var output_shape = [_]usize{ batch_size, total_size };

            // Dealloc previous input and output if they exist
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            // Store input and create output
            self.input = try input.copy();
            self.output = try TensMath.reshape(T, input, output_shape[0..], null);

            return self.output;
        }

        /// Backward pass: Reshape the gradients to the original input shape
        pub fn backward(ctx: *anyopaque, dValues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.original_shape.len == 0) {
                return LayerError.InvalidParameters;
            }

            var dInput = try TensMath.reshape(T, dValues, self.original_shape, null);
            errdefer dInput.deinit();
            return dInput;
        }

        /// Print the flatten layer information
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (choice) {
                0 => std.debug.print("Flatten Layer\n", .{}),
                1 => std.debug.print("Original shape: {any}, Output shape: {any}\n", .{ self.original_shape, self.output.shape }),
                else => {},
            }
        }

        //---------------------------------------------------------------
        //---------------------------- Getters --------------------------
        //---------------------------------------------------------------
        /// Get the number of inputs = product of all dimensions except the first is combined into one
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.original_shape.len < 2) return 0;

            var total: usize = 1;
            for (self.original_shape[1..]) |dim| {
                total *= dim;
            }
            return total;
        }

        /// For Flatten layer, number of neurons = number of inputs after the first dimension
        pub fn get_n_neurons(ctx: *anyopaque) usize {
            return get_n_inputs(ctx);
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.output;
        }
    };
}
