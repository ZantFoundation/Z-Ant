const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

pub const utils = @import("utils_elu.zig");

// --------------------- ELU OPERATOR ---------------------

/// Applies the Elu activation function, allocating a new output tensor.
/// f(x) = alpha * (exp(x) - 1) for x < 0, f(x) = x for x >= 0.
pub fn elu(comptime T: type, input: *const Tensor(T), alpha: T) !Tensor(T) {
    // Validate type
    if (!utils.isFloatType(T)) {
        return TensorMathError.InvalidDataType;
    }

    //validate input is 1D
    const input_shape = input.shape;
    if (input_shape.len != 1) {
        return TensorMathError.InvalidInput;
    }

    //compute output shape
    const output_shape = try utils.get_elu_output_shape(input_shape);
    defer pkg_allocator.free(output_shape);

    // Allocate output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try elu_lean(T, input, &output, alpha);

    return output;
}

/// Applies the Elu activation function in-place on a pre-allocated output tensor.
/// f(x) = alpha * (exp(x) - 1) for x < 0, f(x) = x for x >= 0.
pub fn elu_lean(comptime T: type, input: *const Tensor(T), output: *Tensor(T), alpha: T) !void {
    // Apply Elu element-wise
    const input_data = input.data;
    const output_data = output.data;
    for (input_data, output_data) |x, *y| {
        y.* = if (x < 0) alpha * (std.math.exp(x) - 1.0) else x;
    }
}
