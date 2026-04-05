const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub const utils = @import("utils_log.zig");

//----------------- LOG OPERATOR ------------------------

// Output as a tensor (no lean version)
pub fn log(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    if (!utils.isLogSupportedType(T)) {
        return error.InvalidDataType;
    }

    if (input.data.len == 0) {
        const output_shape = try utils.get_log_output_shape(input.shape);
        defer pkg_allocator.free(output_shape);
        return try Tensor(T).fromShape(&pkg_allocator, output_shape);
    }

    const output_shape = try utils.get_log_output_shape(input.shape);
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
        return error.OutputTensorWrongShape;
    }

    for (input_data, 0..) |x, i| {
        output_data[i] = @log(x);
    }
}
