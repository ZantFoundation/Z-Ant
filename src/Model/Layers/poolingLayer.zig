const std = @import("std");

const zant = @import("../../zant.zig");
const tensor = zant.core.tensor;
const TensMath = zant.core.tensor.math_standard;
const Layer = zant.model.layer;
const LayerError = zant.utils.error_handler.LayerError;
const TensorError = zant.utils.error_handler.TensorError;

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

/// Represents a pooling layer in a neural network.
/// A pooling layer reduces the spatial dimensions of the input tensor
/// while retaining the most significant information, helping to down-sample and reduce computation.
/// This struct supports different pooling types (e.g., max pooling, average pooling).
///
/// @param T The data type of the tensor elements (e.g., `f32`, `f64`, etc.).
/// @note TODO: Padding support is not yet implemented
pub fn PoolingLayer(comptime T: type) type {
    return struct {
        input: tensor.Tensor(T),
        output: tensor.Tensor(T),
        /// A tensor that tracks which elements of the input were used in the pooling operation.
        /// This is typically relevant for max pooling to facilitate backpropagation.
        /// Stored as a tensor of `u8` (binary values).
        used_input: tensor.Tensor(u8),

        /// Pooling Configuration -----------------------
        /// The dimensions of the pooling kernel, specified as `[rows, cols]`.
        /// This defines the size of the pooling window applied to the input tensor.
        kernel: [2]usize,

        /// The stride of the pooling operation, specified as `[rows, cols]`.
        /// This determines the step size of the pooling window as it moves across the input tensor.
        stride: [2]usize,
        poolingType: PoolingType,
        allocator: *const std.mem.Allocator,

        const Self = @This();

        pub fn create(self: *Self) !Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.PoolingLayer,
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

        /// Initialize the layer
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const struct { kernel: [2]usize, stride: [2]usize, poolingType: PoolingType } = @ptrCast(@alignCast(args));

            self.allocator = alloc;

            // Assign kernel and stride arrays directly
            self.kernel = argsStruct.kernel;
            self.stride = argsStruct.stride;
            self.poolingType = argsStruct.poolingType;

            std.debug.print("\nInit Pooling Layer", .{});

            // Only 2D supported
            if (self.kernel.len != 2 or self.stride.len != 2) {
                return LayerError.Only2DSupported;
            }

            // Check Kernel != 0
            for (self.kernel) |val| {
                if (val <= 0) return LayerError.ZeroValueKernel;
            }

            // Check stride != 0
            for (self.stride) |val| {
                if (val <= 0) return LayerError.ZeroValueStride;
            }
        }

        /// Deallocate the layer
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            if (self.used_input.data.len > 0) {
                self.used_input.deinit();
            }

            std.debug.print("\nPooling layer resources deallocated.", .{});
        }

        /// Forward pass of the pooling layer
        pub fn forward(ctx: *anyopaque, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Dealloc previous input if exists
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();
            errdefer self.input.deinit();

            // Dealloc previous output and used_input if they exist
            if (self.output.data.len > 0) {
                self.output.deinit();
            }
            if (self.used_input.data.len > 0) {
                self.used_input.deinit();
            }

            // Call tensor_math's pool_forward function
            const result = try TensMath.pool_forward(T, &self.input, self.kernel, self.stride, self.poolingType);
            self.output = result.output;
            self.used_input = result.used_input;

            return self.output;
        }

        pub fn backward(ctx: *anyopaque, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Call tensor_math's pool_backward function
            return TensMath.pool_backward(T, dValues, self.input.shape, &self.used_input, self.kernel, self.stride);
        }

        /// Print the layer (debug)
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            std.debug.print("\n ************************Pooling layer*********************", .{});
            std.debug.print("\n kernel: {any}  stride:{any}", .{ self.kernel, self.stride });
            if (choice == 0) {
                std.debug.print("\n \n************input", .{});
                self.input.printMultidim();
                std.debug.print("\n \n************output", .{});
                self.output.printMultidim();
            }
            if (choice == 1) {
                std.debug.print("\n   input: [", .{});
                for (0..self.input.shape.len) |i| {
                    std.debug.print("{}", .{self.input.shape[i]});
                    if (i == self.input.shape.len - 1) {
                        std.debug.print("]", .{});
                    } else {
                        std.debug.print(" x ", .{});
                    }
                }
                std.debug.print("\n   output: [", .{});
                for (0..self.output.shape.len) |i| {
                    std.debug.print("{}", .{self.output.shape[i]});
                    if (i == self.output.shape.len - 1) {
                        std.debug.print("]", .{});
                    } else {
                        std.debug.print(" x ", .{});
                    }
                }
                std.debug.print("\n ", .{});
            }
        }

        //---------------------------------------------------------------
        //----------------------------getters----------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            // Return a dummy value since not supported
            _ = ctx;
            return 0;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            // Return a dummy value since not supported
            _ = ctx;
            return 0;
        }

        pub fn get_input(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.output;
        }
    };
}
