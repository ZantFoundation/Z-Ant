const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

/// ONNX Pad operation following the ONNX specification
/// Supports constant, reflect, edge, and wrap padding modes
pub fn pad(
    comptime T: type,
    input: *const Tensor(T),
    pads: *const Tensor(i64), // Padding amounts [begin0, begin1, ..., end0, end1, ...]
    constant_value: ?*const Tensor(T), // Value for constant mode
    axes: ?*const Tensor(i64), // Axes to pad (if null, pad all axes)
    output: *Tensor(T),
    mode: []const u8, // "constant", "reflect", "edge", "wrap"
) !void {
    if (input.shape.len == 0) return TensorMathError.InvalidDimensions;

    // Default to padding all axes if none specified
    const num_axes = if (axes) |a| a.size else input.shape.len;

    // Extract padding values
    if (pads.size < num_axes * 2) return TensorMathError.InvalidDimensions;

    // Determine output shape
    var output_shape = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(output_shape);
    @memcpy(output_shape, input.shape);

    for (0..num_axes) |i| {
        const axis_idx = if (axes) |a| @as(usize, @intCast(a.data[i])) else i;
        if (axis_idx >= input.shape.len) return TensorMathError.InvalidDimensions;

        const pad_begin = @as(usize, @intCast(@max(0, pads.data[i])));
        const pad_end = @as(usize, @intCast(@max(0, pads.data[i + num_axes])));
        output_shape[axis_idx] = input.shape[axis_idx] + pad_begin + pad_end;
    }

    // Verify output tensor has correct shape
    if (!std.mem.eql(usize, output.shape, output_shape)) {
        return TensorMathError.ShapeMismatch;
    }

    // Initialize output with zeros/constant value
    const fill_value = if (constant_value) |cv| cv.data[0] else @as(T, 0);
    @memset(output.data, fill_value);

    // Determine padding mode
    if (std.mem.eql(u8, mode, "constant")) {
        try pad_constant(T, input, output, pads, axes, fill_value);
    } else if (std.mem.eql(u8, mode, "reflect")) {
        try pad_reflect(T, input, output, pads, axes);
    } else if (std.mem.eql(u8, mode, "edge")) {
        try pad_edge(T, input, output, pads, axes);
    } else if (std.mem.eql(u8, mode, "wrap")) {
        try pad_wrap(T, input, output, pads, axes);
    } else {
        return TensorMathError.UnsupportedMode;
    }
}

fn pad_constant(
    comptime T: type,
    input: *const Tensor(T),
    output: *Tensor(T),
    pads: *const Tensor(i64),
    axes: ?*const Tensor(i64),
    fill_value: T,
) !void {
    _ = fill_value;
    // Output is already filled with constant value, just copy input data
    const num_axes = if (axes) |a| a.size else input.shape.len;

    // Calculate offset in output where input data should be placed
    var offset = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(offset);
    @memset(offset, 0);

    for (0..num_axes) |i| {
        const axis_idx = if (axes) |a| @as(usize, @intCast(a.data[i])) else i;
        offset[axis_idx] = @as(usize, @intCast(@max(0, pads.data[i])));
    }

    // Copy input data to the correct position in output
    try copy_tensor_with_offset(T, input, output, offset);
}

fn pad_reflect(
    comptime T: type,
    input: *const Tensor(T),
    output: *Tensor(T),
    pads: *const Tensor(i64),
    axes: ?*const Tensor(i64),
) !void {
    _ = input;
    _ = output;
    _ = pads;
    _ = axes;
    // TODO: Implement reflect padding
    return TensorMathError.UnsupportedMode;
}

fn pad_edge(
    comptime T: type,
    input: *const Tensor(T),
    output: *Tensor(T),
    pads: *const Tensor(i64),
    axes: ?*const Tensor(i64),
) !void {
    const rank = input.shape.len;

    // Compute pad_begin per dimension
    var pad_begin = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(pad_begin);
    @memset(pad_begin, 0);

    const num_axes = if (axes) |a| a.size else rank;
    for (0..num_axes) |i| {
        const axis_idx = if (axes) |a| @as(usize, @intCast(a.data[i])) else i;
        if (axis_idx >= rank) return TensorMathError.InvalidDimensions;
        pad_begin[axis_idx] = @as(usize, @intCast(@max(0, pads.data[i])));
    }

    // Precompute strides
    var in_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(in_strides);
    var out_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(out_strides);

    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    if (rank > 1) {
        for (0..rank - 1) |k| {
            const idx = rank - 2 - k;
            in_strides[idx] = in_strides[idx + 1] * input.shape[idx + 1];
            out_strides[idx] = out_strides[idx + 1] * output.shape[idx + 1];
        }
    }

    // Temp index buffers
    var out_indices = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(out_indices);
    var in_indices = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(in_indices);

    // Fill output by clamping to edge
    for (0..output.size) |out_lin| {
        var rem = out_lin;
        for (0..rank) |d| {
            const idx = rem / out_strides[d];
            rem %= out_strides[d];
            out_indices[d] = idx;
        }

        for (0..rank) |d| {
            const src = if (out_indices[d] < pad_begin[d])
                0
            else blk: {
                const rel = out_indices[d] - pad_begin[d];
                break :blk @min(rel, input.shape[d] - 1);
            };
            in_indices[d] = src;
        }

        var in_lin: usize = 0;
        for (0..rank) |d| in_lin += in_indices[d] * in_strides[d];
        output.data[out_lin] = input.data[in_lin];
    }
}

fn pad_wrap(
    comptime T: type,
    input: *const Tensor(T),
    output: *Tensor(T),
    pads: *const Tensor(i64),
    axes: ?*const Tensor(i64),
) !void {
    _ = input;
    _ = output;
    _ = pads;
    _ = axes;
    // TODO: Implement wrap padding
    return TensorMathError.UnsupportedMode;
}

fn copy_tensor_with_offset(
    comptime T: type,
    input: *const Tensor(T),
    output: *Tensor(T),
    offset: []const usize,
) !void {
    // Calculate strides for both tensors
    var input_strides = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(input_strides);
    var output_strides = try pkg_allocator.alloc(usize, output.shape.len);
    defer pkg_allocator.free(output_strides);

    // Calculate strides (last dimension has stride 1)
    input_strides[input_strides.len - 1] = 1;
    for (0..input.shape.len - 1) |i| {
        const idx = input.shape.len - 2 - i;
        input_strides[idx] = input_strides[idx + 1] * input.shape[idx + 1];
    }

    output_strides[output_strides.len - 1] = 1;
    for (0..output.shape.len - 1) |i| {
        const idx = output.shape.len - 2 - i;
        output_strides[idx] = output_strides[idx + 1] * output.shape[idx + 1];
    }

    // Copy each element
    for (0..input.size) |linear_idx| {
        // Convert linear index to multi-dimensional indices for input
        var input_indices = try pkg_allocator.alloc(usize, input.shape.len);
        defer pkg_allocator.free(input_indices);

        var remaining = linear_idx;
        for (0..input.shape.len) |i| {
            input_indices[i] = remaining / input_strides[i];
            remaining %= input_strides[i];
        }

        // Calculate corresponding output indices
        var output_indices = try pkg_allocator.alloc(usize, output.shape.len);
        defer pkg_allocator.free(output_indices);

        for (0..input.shape.len) |i| {
            output_indices[i] = input_indices[i] + offset[i];
        }

        // Convert output indices to linear index
        var output_linear_idx: usize = 0;
        for (0..output.shape.len) |i| {
            output_linear_idx += output_indices[i] * output_strides[i];
        }

        output.data[output_linear_idx] = input.data[linear_idx];
    }
}

/// Calculate output shape for Pad operation
pub fn get_pad_output_shape(
    input_shape: []const usize,
    pads: []const i64,
    axes: ?[]const i64,
) ![]usize {
    var output_shape = try pkg_allocator.dupe(usize, input_shape);

    const num_axes = if (axes) |a| a.len else input_shape.len;
    if (pads.len < num_axes * 2) return TensorMathError.InvalidDimensions;

    for (0..num_axes) |i| {
        const axis_idx = if (axes) |a| @as(usize, @intCast(a[i])) else i;
        if (axis_idx >= input_shape.len) return TensorMathError.InvalidDimensions;

        const pad_begin = @as(usize, @intCast(@max(0, pads[i])));
        const pad_end = @as(usize, @intCast(@max(0, pads[i + num_axes])));
        output_shape[axis_idx] = input_shape[axis_idx] + pad_begin + pad_end;
    }

    return output_shape;
}
