const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator.allocator;

const utils_oneHot = @import("utils_oneHot.zig");
const get_onehot_output_shape = utils_oneHot.get_one_hot_output_shape;

pub fn one_hot(comptime T: type, indices: *const Tensor(i64), depth: *const Tensor(i64), values: *const Tensor(T), axis: i64) !Tensor(T) {
    // Check types
    const allowed_types = [_]type{
        f32,  f64,
        bool, i8,
        i16,  i32,
        i64,  u8,
        u16,  u32,
        u64,
    };

    var valid_type = false;
    inline for (allowed_types) |Allowed| {
        if (T == Allowed) {
            valid_type = true;
            break;
        }
    }
    if (!valid_type) {
        return error.InvalidDataType;
    }

    // Check depth (scalar or rank 1 with one element)
    if (depth.shape.len > 1 or (depth.shape.len == 1 and depth.shape[0] != 1)) {
        return error.InvalidDepthShape;
    }
    const depth_val = depth.data[0];
    if (depth_val <= 0) {
        return error.InvalidDepthValue;
    }

    // Check values (rank 1 with 2 elements)
    if (values.shape.len != 1 or values.shape[0] != 2) {
        return error.InvalidValuesShape;
    }

    // Compute the output shape
    const output_shape = try get_onehot_output_shape(indices.shape, depth_val, axis);
    var output = try Tensor(T).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();
    defer pkgAllocator.free(output_shape);

    // Call the lean version
    try one_hot_lean(T, indices, depth_val, values, axis, &output);

    return output;
}

pub fn one_hot_lean(comptime T: type, indices: *const Tensor(i64), depth: i64, values: *const Tensor(T), axis: i64, output: *Tensor(T)) !void {
    // Initialize output with off_value
    for (output.data) |*val| {
        val.* = values.data[0]; // off_value
    }

    // Normalize axis
    const output_rank = @as(i64, @intCast(output.shape.len));
    const normalized_axis = if (axis < 0) axis + output_rank else axis;

    // Validate normalized axis
    if (normalized_axis < 0 or normalized_axis >= output_rank) {
        return error.InvalidAxis;
    }

    // Iterate over all indices
    const total_elements = blk: {
        var prod: usize = 1;
        for (indices.shape) |dim| prod *= dim;
        break :blk prod;
    };

    for (0..total_elements) |flat_idx| {
        const index_val = indices.data[flat_idx];

        // Skip indices out of range
        if (index_val < -depth or index_val >= depth) {
            continue;
        }

        // Convert negative indices to positive
        const index = if (index_val < 0) index_val + depth else index_val;

        // Calculate multi-dimensional coordinates from flat index
        var input_coords = try pkgAllocator.alloc(usize, indices.shape.len);
        defer pkgAllocator.free(input_coords);

        var temp_idx = flat_idx;
        var i: usize = indices.shape.len;
        while (i > 0) {
            i -= 1;
            input_coords[i] = temp_idx % indices.shape[i];
            temp_idx /= indices.shape[i];
        }

        // Build output coordinates by inserting the new axis
        var output_coords = try pkgAllocator.alloc(usize, output.shape.len);
        defer pkgAllocator.free(output_coords);

        // Copy coordinates before the new axis
        for (0..@intCast(normalized_axis)) |coord_idx| {
            output_coords[coord_idx] = input_coords[coord_idx];
        }

        // Set the coordinate for the new axis (depth dimension)
        output_coords[@intCast(normalized_axis)] = @intCast(index);

        // Copy coordinates after the new axis
        for (@intCast(normalized_axis)..input_coords.len) |coord_idx| {
            output_coords[coord_idx + 1] = input_coords[coord_idx];
        }

        // Set on_value at the calculated position
        const output_idx = try output.get_flat_index(output_coords);
        output.data[output_idx] = values.data[1]; // on_value
    }
}

// Alternative implementation that might be clearer and more efficient
pub fn onehot_lean_v2(comptime T: type, indices: *const Tensor(i64), depth: i64, values: *const Tensor(T), axis: i64, output: *Tensor(T)) !void {
    // Initialize output with off_value
    @memset(output.data, values.data[0]);

    // Normalize axis
    const output_rank = @as(i64, @intCast(output.shape.len));
    const normalized_axis = if (axis < 0) axis + output_rank else axis;

    // Validate normalized axis
    if (normalized_axis < 0 or normalized_axis >= output_rank) {
        return error.InvalidAxis;
    }

    const axis_usize = @as(usize, @intCast(normalized_axis));

    // Calculate strides for output tensor
    var output_strides = try pkgAllocator.alloc(usize, output.shape.len);
    defer pkgAllocator.free(output_strides);

    output_strides[output.shape.len - 1] = 1;
    if (output.shape.len > 1) {
        var i: usize = output.shape.len - 1;
        while (i > 0) {
            i -= 1;
            output_strides[i] = output_strides[i + 1] * output.shape[i + 1];
        }
    }

    // Process each element in the indices tensor
    const total_elements = blk: {
        var prod: usize = 1;
        for (indices.shape) |dim| prod *= dim;
        break :blk prod;
    };

    for (0..total_elements) |flat_idx| {
        const index_val = indices.data[flat_idx];

        // Skip indices out of range
        if (index_val < -depth or index_val >= depth) {
            continue;
        }

        // Convert negative indices to positive
        const index = if (index_val < 0) index_val + depth else index_val;

        // Calculate position in output tensor
        var output_idx: usize = 0;
        var temp_idx = flat_idx;

        // Map indices coordinates to output coordinates
        var coord_idx: usize = 0;
        for (0..output.shape.len) |out_dim| {
            if (out_dim == axis_usize) {
                // This is the new axis - use the index value
                output_idx += @as(usize, @intCast(index)) * output_strides[out_dim];
            } else {
                // This corresponds to an original dimension
                const coord = temp_idx % indices.shape[coord_idx];
                temp_idx /= indices.shape[coord_idx];
                output_idx += coord * output_strides[out_dim];
                coord_idx += 1;
            }
        }

        // Set on_value
        output.data[output_idx] = values.data[1];
    }
}
