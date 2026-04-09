const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub const utils = @import("utils_exp.zig");

// --------------------- EXP OPERATOR ---------------------

/// Applies the exponential function element-wise, allocating a new output tensor.
/// f(x) = exp(x)
pub fn exp(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    // Validate type
    if (!utils.isFloatType(T)) {
        return error.InvalidDataType;
    }

    if (input.data.len == 0) {
        return error.ZeroSizeTensor;
    }

    // Compute output shape
    const output_shape = try utils.get_exp_output_shape(input.shape);
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
        return error.OutputTensorWrongShape;
    }

    for (input_data, output_data) |x, *y| {
        y.* = @exp(x);
    }
}
