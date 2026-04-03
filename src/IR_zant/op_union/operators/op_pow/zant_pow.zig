const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_pow_output_shape(comptime T: type, comptime T1: type, base: *const Tensor(T), exp: *const Tensor(T1)) ![]usize {

    //broadcast
    const len1 = base.shape.len;
    const len2 = exp.shape.len;
    const maxLen = @max(len1, len2);

    //creating the output
    const output = try pkg_allocator.alloc(usize, maxLen);
    errdefer pkg_allocator.free(output);

    //setting offsets
    const offset1: usize = maxLen - len1;
    const offset2: usize = maxLen - len2;

    //filling output shape
    var pos: usize = 0;
    while (pos < maxLen) : (pos += 1) {
        const dim1: usize = if (pos < offset1) 1 else base.shape[pos - offset1];
        const dim2: usize = if (pos < offset2) 1 else exp.shape[pos - offset2];

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }

        output[pos] = if (dim1 >= dim2) dim1 else dim2;
    }

    return output;
}

pub fn pow(comptime T: type, comptime T1: type, base: *Tensor(T), exp: *Tensor(T1)) !Tensor(T) {

    //check for unsupported types at compile time
    comptime {
        const isSupported = switch (T) {
            f16, f32, f64, i32, i64 => true,
            else => false,
        };
        const isSupported2 = switch (T1) {
            f16, f32, f64, i32, i64 => true,
            else => false,
        };
        if (!isSupported or !isSupported2) return TensorMathError.InvalidDataType;
    }

    const outputShape = try get_pow_output_shape(T, T1, base, exp);
    defer pkg_allocator.free(outputShape);

    var output = try Tensor(T).fromShape(&pkg_allocator, outputShape);
    errdefer output.deinit();

    try pow_lean(T, T1, base, exp, &output);

    return output;
}

pub fn pow_lean(comptime T: type, comptime T1: type, baseTensor: *Tensor(T), expTensor: *Tensor(T1), output: *Tensor(T)) !void {
    for (0..output.size) |idx| {
        const coords = try indexToCoords(idx, output.shape);
        defer pkg_allocator.free(coords);

        const base_idx = getBroadcastIndex(coords, baseTensor.shape, output.shape);
        const exp_idx = getBroadcastIndex(coords, expTensor.shape, output.shape);

        const baseValue = baseTensor.data[base_idx];
        const expValueCasted = castToType(T, T1, expTensor.data[exp_idx]);

        if (baseValue == 0 and expValueCasted < 0) return TensorMathError.DivisionError;

        const result = std.math.pow(T, baseValue, expValueCasted);

        output.data[idx] = result;
    }
}

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
inline fn product(slice: []const usize) usize {
    var result: usize = 1;
    for (slice) |val| {
        result *= val;
    }
    return result;
}

//used to cast the element in the pow op
inline fn castToType(comptime TargetType: type, comptime SourceType: type, value: SourceType) TargetType {
    const target_info = @typeInfo(TargetType);
    const source_info = @typeInfo(SourceType);

    if (target_info == .float and source_info == .float) {
        return @floatCast(value);
    } else if (target_info == .float and source_info == .int) {
        return @floatFromInt(value);
    } else if (target_info == .int and source_info == .float) {
        return @intFromFloat(value);
    } else if (target_info == .int and source_info == .int) {
        return @intCast(value);
    } else {
        @compileError("Unsupported type conversion");
    }
}
