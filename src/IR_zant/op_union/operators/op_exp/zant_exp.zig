const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- EXP OPERATOR ---------------------

/// Computes the output shape for the Exp operator.
/// Returns a slice with the same shape as the input, as Exp is an element-wise operation.
pub fn get_exp_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

/// Applies the exponential function element-wise, allocating a new output tensor.
/// f(x) = exp(x)
pub fn exp(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    // Validate type
    if (!isFloatType(T)) {
        return TensorMathError.InvalidDataType;
    }

    if (input.data.len == 0) {
        return TensorError.ZeroSizeTensor;
    }

    // Compute output shape
    const output_shape = try get_exp_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    // Allocate output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try exp_lean(T, input, &output);

    return output;
}

/// Applies the exponential function element-wise on a pre-allocated output tensor.
/// f(x) = exp(x)
pub fn exp_lean(comptime T: type, input: *const Tensor(T), output: *Tensor(T)) !void {
    // Apply exp element-wise
    const input_data = input.data;
    const output_data = output.data;

    if (input_data.len != output_data.len) {
        return TensorError.OutputTensorWrongShape;
    }

    for (input_data, output_data) |x, *y| {
        y.* = @exp(x);
    }
}

/// Helper function to check if T is a supported float type.
fn isFloatType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Float => true,
        else => false,
    };
}
