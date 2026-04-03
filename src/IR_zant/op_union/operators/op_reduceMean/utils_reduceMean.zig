const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_reduce_mean_output_shape(inputs: []const []usize) ![]usize {
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
pub fn getBroadcastIndex(output_coords: []const usize, input_shape: []const usize, output_shape: []const usize) usize {
    std.debug.assert(output_coords.len == output_shape.len);
    std.debug.assert(input_shape.len <= output_shape.len);

    const rank_diff = output_shape.len - input_shape.len;
    var input_index: usize = 0;

    for (output_shape, output_coords, 0..) |_, coord, i| {
        if (i < rank_diff) continue;

        const input_dim_idx = i - rank_diff;
        const in_dim = input_shape[input_dim_idx];

        const effective_coord = if (in_dim == 1) 0 else coord;
        std.debug.assert(effective_coord < in_dim);

        var stride: usize = 1;
        for (input_shape[input_dim_idx + 1 ..]) |dim| {
            stride *= dim;
        }
        input_index += effective_coord * stride;
    }

    return input_index;
}

/// Converts a linear index to multidimensional coordinates based on the tensor shape.
pub fn indexToCoords(index: usize, shape: []const usize) ![]usize {
    if (index >= product(shape)) {
        return error.IndexOutOfBounds;
    }

    var coords = try pkg_allocator.alloc(usize, shape.len);
    errdefer pkg_allocator.free(coords);

    var remaining = index;
    for (shape, 0..) |_, i| {
        if (i == shape.len - 1) {
            coords[i] = remaining;
        } else {
            const stride = product(shape[i + 1 ..]);
            coords[i] = remaining / stride;
            remaining = remaining % stride;
        }
    }

    return coords;
}

/// Converts multidimensional coordinates to a linear index based on the tensor shape.
pub fn coordsToIndex(coords: []const usize, shape: []const usize) usize {
    std.debug.assert(coords.len == shape.len);

    var index: usize = 0;
    for (shape, coords, 0..) |dim, coord, i| {
        std.debug.assert(coord < dim);
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
