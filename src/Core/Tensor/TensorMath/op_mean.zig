const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const TensMath = @import("tensor_math_standard.zig");
const op_mat_mul = @import("op_mat_mul.zig");

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

pub fn mean_standard(comptime T: anytype, inputs: []*Tensor(T)) !Tensor(T) {
    if (inputs.len == 0) {
        return TensorMathError.EmptyTensorList;
    }

    const type_info = @typeInfo(T);
    if (type_info != .float or (T != f32 and T != f64 and T != f16)) {
        return TensorMathError.InvalidDataType;
    }
    for (inputs) |tensor| {
        if (@TypeOf(tensor.data) != @TypeOf(inputs[0].data)) {
            return TensorMathError.MismatchedDataTypes;
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

    try mean_lean(T, inputs, &output);

    return output;
}

pub inline fn mean_lean(comptime T: anytype, inputs: []*Tensor(T), output: *Tensor(T)) !void {
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

pub fn get_mean_output_shape(inputs: []const []usize) ![]usize {
    if (inputs.len == 0) {
        return TensorMathError.EmptyTensorList;
    }

    var max_rank: usize = 0;
    for (inputs) |shape| {
        if (shape.len > max_rank) {
            max_rank = shape.len;
        }
    }

    var output_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(output_shape);
    @memset(output_shape, 1);

    for (inputs) |shape| {
        const rank_diff = max_rank - shape.len;
        for (shape, 0..) |dim, i| {
            const out_idx = rank_diff + i;
            const current_out_dim = output_shape[out_idx];

            if (dim != current_out_dim) {
                if (current_out_dim == 1) {
                    output_shape[out_idx] = dim;
                } else if (dim != 1) {
                    pkg_allocator.free(output_shape);
                    return TensorMathError.IncompatibleBroadcastShapes;
                }
            }
        }
    }

    return output_shape;
}

/// Computes the linear index in an input tensor given a position in the output, applying broadcasting.
/// - `output_coords`: Multidimensional coordinates in the output tensor.
/// - `input_shape`: Shape of the input tensor.
/// - `output_shape`: Shape of the output tensor (for alignment).
/// Returns the linear index in the input tensor.
pub fn getBroadcastIndex(output_coords: []const usize, input_shape: []const usize, output_shape: []const usize) usize {
    std.debug.assert(output_coords.len == output_shape.len); // Coordinates must match the output
    std.debug.assert(input_shape.len <= output_shape.len); // Input may have fewer dimensions

    // Compute the offset to align shapes from the right
    const rank_diff = output_shape.len - input_shape.len;
    var input_index: usize = 0;

    // Iterate over the output dimensions
    for (output_shape, output_coords, 0..) |_, coord, i| {
        // If we are beyond the input dimensions, they do not contribute to the index
        if (i < rank_diff) continue;

        // Corresponding index in the input shape
        const input_dim_idx = i - rank_diff;
        const in_dim = input_shape[input_dim_idx];

        // Broadcasting: if the input dimension is 1, use 0; otherwise use the coordinate
        const effective_coord = if (in_dim == 1) 0 else coord;
        std.debug.assert(effective_coord < in_dim); // Verify that the coordinate is valid

        // Compute the contribution to the linear index
        var stride: usize = 1;
        for (input_shape[input_dim_idx + 1 ..]) |dim| {
            stride *= dim;
        }
        input_index += effective_coord * stride;
    }

    return input_index;
}

/// Converts a linear index to multidimensional coordinates based on the tensor shape.
/// - `index`: Linear index (0-based).
/// - `shape`: Shape of the tensor.
/// Returns an array of coordinates (must be freed by the caller).
pub fn indexToCoords(index: usize, shape: []const usize) ![]usize {
    if (index >= product(shape)) {
        return error.IndexOutOfBounds;
    }

    var coords = try pkg_allocator.alloc(usize, shape.len);
    errdefer pkg_allocator.free(coords);

    var remaining = index;
    for (shape, 0..) |_, i| {
        if (i == shape.len - 1) {
            coords[i] = remaining; // Last dimension: direct remainder
        } else {
            const stride = product(shape[i + 1 ..]); // Product of the subsequent dimensions
            coords[i] = remaining / stride;
            remaining = remaining % stride;
        }
    }

    return coords;
}

/// Converts multidimensional coordinates to a linear index based on the tensor shape.
/// - `coords`: Multidimensional coordinates.
/// - `shape`: Shape of the tensor.
/// Returns the corresponding linear index.
pub fn coordsToIndex(coords: []const usize, shape: []const usize) usize {
    std.debug.assert(coords.len == shape.len); // Coordinates must match the shape

    var index: usize = 0;
    for (shape, coords, 0..) |dim, coord, i| {
        std.debug.assert(coord < dim); // Verify that the coordinate is valid
        const stride = if (i == shape.len - 1) 1 else product(shape[i + 1 ..]);
        index += coord * stride;
    }

    return index;
}

/// Computes the product of an array of usize.
fn product(slice: []const usize) usize {
    var result: usize = 1;
    for (slice) |val| {
        result *= val;
    }
    return result;
}
