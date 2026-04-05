const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;

const pkg_allocator = zant.utils.allocator.allocator;

const get_unsqueeze_output_shape = @import("utils_unsqueeze.zig").get_unsqueeze_output_shape;

/// Implements https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
/// Insert single-dimensional entries into the shape of the data tensor.
pub fn unsqueeze(comptime T: type, data: *Tensor(T), axes: *Tensor(i64)) !Tensor(T) {

    // Output rank
    const out_rank = data.shape.len + axes.size;
    const conv_out_rank: i64 = @intCast(out_rank);

    for (0..axes.data.len) |i| {

        // Check if axes are within bounds
        if (axes.data[i] < -conv_out_rank or axes.data[i] >= out_rank) {
            return error.AxisOutOfBounds;
        }

        // Check for duplicates
        for (0..i) |j| {
            if (axes.data[i] == axes.data[j]) {
                return error.DuplicateAxis;
            }
        }
    }

    const output_shape = try get_unsqueeze_output_shape(data.shape, axes.data);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(data.allocator, output_shape);

    try unsqueeze_lean(T, data, axes, &output);

    return output;
}

/// Lean version of unsqueeze, note that previous information stored in output tensor is lost
pub fn unsqueeze_lean(comptime T: type, input: *Tensor(T), axes: *Tensor(i64), output: *Tensor(T)) !void {
    _ = axes;
    @memcpy(output.data, input.data);
}
