const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- MIN OPERATOR ---------------------

const utils_min = @import("utils_min.zig");
const get_min_output_shape = utils_min.get_min_output_shape;

/// Applies the min function element-wise across multiple tensors, allocating a new output tensor.
pub fn min(comptime T: anytype, inputs: []*Tensor(T)) !Tensor(T) {
    if (inputs.len == 0) return error.InvalidDimensions;
    if (inputs.len == 1) return inputs[0].copy();

    // For now, assume all tensors have the same shape
    const output_shape = try get_min_output_shape(&[_][]const usize{inputs[0].shape});
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(inputs[0].allocator, output_shape);
    errdefer output.deinit();

    try min_lean(T, inputs, &output);
    return output;
}

/// Lean version that computes min in-place using a pre-allocated output tensor.
pub fn min_lean(comptime T: anytype, inputs: []*Tensor(T), output: *Tensor(T)) !void {
    if (inputs.len == 0) return error.InvalidDimensions;
    if (inputs.len == 1) {
        @memcpy(output.data, inputs[0].data);
        return;
    }

    // Start with the first tensor
    @memcpy(output.data, inputs[0].data);

    // Apply min with each subsequent tensor
    for (inputs[1..]) |input| {
        if (input.size != output.size) return error.InvalidDimensions;

        for (0..output.size) |i| {
            if (input.data[i] < output.data[i]) {
                output.data[i] = input.data[i];
            }
        }
    }
}

