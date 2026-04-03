const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

pub fn leaky_relu_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T), slope: T) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply Leaky ReLU derivative: f'(x) = 1 if x > 0, slope if x <= 0
    for (0..gradient.size) |i| {
        gradient.data[i] *= if (act_relu_input.data[i] > 0) 1 else slope;
    }
}

pub fn get_leaky_relu_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}
