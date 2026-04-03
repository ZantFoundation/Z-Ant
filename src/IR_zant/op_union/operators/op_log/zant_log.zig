const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

//----------------- LOG OPERATOR ------------------------

pub fn get_log_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

// Output as a tensor (no lean version)
pub fn log(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    if (!isLogSupportedType(T)) {
        return TensorMathError.InvalidDataType;
    }

    if (input.data.len == 0) {
        const output_shape = try get_log_output_shape(input.shape);
        defer pkg_allocator.free(output_shape);
        return try Tensor(T).fromShape(&pkg_allocator, output_shape);
    }

    const output_shape = try get_log_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    var outputTensor = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer outputTensor.deinit();

    try log_lean(T, input, &outputTensor);

    return outputTensor;
}

//output void (lean version)
pub inline fn log_lean(comptime T: type, input: *const Tensor(T), output: *Tensor(T)) !void {
    const input_data = input.data;
    const output_data = output.data;

    if (input_data.len != output_data.len) {
        return TensorError.OutputTensorWrongShape;
    }

    for (input_data, 0..) |x, i| {
        output_data[i] = @log(x);
    }
}

// Check whether the tensor type is supported or not
fn isLogSupportedType(comptime T: type) bool {
    return switch (T) {
        f16, f32, f64 => true,
        else => false,
    };
}
