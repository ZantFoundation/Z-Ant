const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;
const TensMath = IR_zant.core.math_standard;
const utils_reduceMean = @import("utils_reduceMean.zig");
const get_mean_output_shape = utils_reduceMean.get_reduce_mean_output_shape;
const indexToCoords = utils_reduceMean.indexToCoords;
const getBroadcastIndex = utils_reduceMean.getBroadcastIndex;

// The ONNX mean operation computes the element-wise average of a
// variable-length list of input tensors, supporting from 1 to a theoretically
// unlimited number of tensors, and produces a single output tensor with the same
// data type as the inputs (e.g. float or double). It does so by aligning the tensors
// through NumPy-style multidirectional broadcasting, which handles different shapes
// by virtually expanding smaller dimensions (when compatible, i.e. equal or equal to 1)
// to match the larger ones, without physically modifying the data. For each position in the output,
// it sums the corresponding values of the inputs (mapped via broadcasting) and divides by the number
// of tensors, yielding a result that reflects the arithmetic mean across all inputs,
// with a shape inferred as the maximum of the compatible dimensions."

pub fn reduce_mean(comptime T: anytype, inputs: []*Tensor(T)) !Tensor(T) {
    if (inputs.len == 0) {
        return error.EmptyTensorList;
    }

    const type_info = @typeInfo(T);
    if (type_info != .float or (T != f32 and T != f64 and T != f16)) {
        return error.InvalidDataType;
    }
    for (inputs) |tensor| {
        if (@TypeOf(tensor.data) != @TypeOf(inputs[0].data)) {
            return error.MismatchedDataTypes;
        }
    }

    var input_shapes = try pkg_allocator.alloc([]usize, inputs.len);
    defer pkg_allocator.free(input_shapes);
    for (inputs, 0..) |tensor, i| {
        input_shapes[i] = tensor.shape;
    }

    const output_shape = try get_mean_output_shape(input_shapes);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try reduce_mean_lean(T, inputs, &output);

    return output;
}

pub inline fn reduce_mean_lean(comptime T: anytype, inputs: []*Tensor(T), output: *Tensor(T)) !void {
    // Iterate over every position in the output
    for (0..output.size) |idx| {
        // Convert linear index to multidimensional coordinates
        const coords = indexToCoords(idx, output.shape) catch unreachable; // Errore gestito in mean_standard
        defer pkg_allocator.free(coords);

        // Compute the sum of input values for this position
        var sum: T = 0;
        for (inputs) |tensor| {
            const input_idx = getBroadcastIndex(coords, tensor.shape, output.shape);
            sum += tensor.data[input_idx];
        }

        // Compute the mean and write to the output tensor
        output.data[idx] = sum / @as(T, @floatFromInt(inputs.len));
    }
}
