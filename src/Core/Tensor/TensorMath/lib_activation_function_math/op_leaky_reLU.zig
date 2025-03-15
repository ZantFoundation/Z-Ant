const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

/// Leaky ReLU is a variant of ReLU that allows a small, positive gradient when the input is negative.
/// This can help prevent the dying ReLU problem, where neurons stop learning if they get stuck in the negative side of the ReLU function.
/// The leaky parameter is a small value that determines how much the function leaks into the negative side.
/// A common value for the leaky parameter is 0.01.
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0
/// f(x) = alpha * x if x <= 0
/// where alpha is a small constant.
/// The derivative of the Leaky ReLU function is:
/// f'(x) = 1 if x > 0
/// f'(x) = alpha if x <= 0
pub inline fn leakyReLU(comptime T: anytype, tensor: *Tensor(T), slope: T) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try lean_leakyReLU(T, tensor, slope, &output_tensor);

    return output_tensor;
}

pub inline fn lean_leakyReLU(comptime T: anytype, input_tensor: *Tensor(T), slope: T, output_tensor: *Tensor(T)) !void {
    //apply Leaky ReLU suing relu self.relu() - (-neg_slope*self).relu()
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] <= 0) {
            output_tensor.data[i] = slope * input_tensor.data[i];
        } else {
            output_tensor.data[i] = input_tensor.data[i];
        }
    }
}

pub fn leakyReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T), slope: T) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply Leaky ReLU derivative: f'(x) = 1 if x > 0, slope if x <= 0
    for (0..gradient.size) |i| {
        gradient.data[i] *= if (act_relu_input.data[i] > 0) 1 else slope;
    }
}
