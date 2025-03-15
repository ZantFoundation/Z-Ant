const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

/// ReLU (Rectified Linear Unit).
/// It outputs the input directly if it's positive, but returns zero for any negative input.
pub inline fn ReLU_standard(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;
    try lean_ReLU(T, tensor, &output);

    return output;
}

pub inline fn lean_ReLU(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    //apply ReLU
    //OSS: can be improved, see how did I parallelized CPU Tensor Sum
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] > 0) output_tensor.data[i] = input_tensor.data[i];
    }
}

pub inline fn ReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T)) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply ReLU derivative: f'(x) = 0 if x <= 0, 1 if x > 0
    for (0..gradient.size) |i| {
        if (act_relu_input.data[i] <= 0) {
            gradient.data[i] = 0;
        }
        // else gradient remains unchanged since f'(x) = 1 for x > 0
    }
}
