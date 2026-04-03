const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

pub fn sigmoid_backward(comptime T: anytype, gradient: *Tensor(T), act_forward_out: *Tensor(T)) !void {
    //checks
    if (gradient.size <= 0 or act_forward_out.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_forward_out.size) return TensorMathError.InputTensorDifferentSize;

    //apply Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    for (0..gradient.size) |i| {
        const sigmoid_output = act_forward_out.data[i];
        gradient.data[i] *= sigmoid_output * (1 - sigmoid_output);
    }
}

pub fn get_sigmoid_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}
