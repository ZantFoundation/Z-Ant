const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

const get_squeeze_output_shape = @import("utils_squeeze.zig").get_squeeze_output_shape;

// Squeeze - 23 : https://onnx.ai/onnx/operators/onnx__Squeeze.html
// Remove single-dimensional entries from the shape of a tensor
// Takes an input axes with a list of axes to squeeze
// If axes is not provided, all the single dimensions will be removed from the shape
// If an axis is selected with shape entry not equal to one, an error is raised

pub fn squeeze_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    @memcpy(output.data, input.data);
}

pub fn squeeze(comptime T: anytype, input: *Tensor(T), axes: ?[]const i64) !Tensor(T) {
    const output_shape = try get_squeeze_output_shape(input.shape, axes);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(input.allocator, output_shape);
    errdefer output.deinit();

    try squeeze_lean(T, input, &output);

    return output;
}
