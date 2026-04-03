const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

const get_slice_output_shape = @import("utils_slice.zig").get_slice_output_shape;

/// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
/// Takes a tensor and extracts a slice along multiple axes.
/// starts: Starting indices for each axis as a tensor
/// ends: Ending indices for each axis (exclusive) as a tensor
/// axes: Which axes to slice (if null, assumes [0,1,2,...]) as a tensor
/// steps: Step sizes for each axis (if null, assumes all 1s) as a tensor
pub fn slice(comptime T: type, comptime T1: type, input: *Tensor(T), starts: *Tensor(T1), ends: *Tensor(T1), axes: ?*Tensor(T1), steps: ?*Tensor(T1)) !Tensor(T) {
    std.log.debug("\n[DEBUG] slice called", .{});

    // Convert tensor inputs to arrays for easier processing
    const starts_array = try tensorToI64Array(T1, starts);
    defer pkg_allocator.free(starts_array);

    const ends_array = try tensorToI64Array(T1, ends);
    defer pkg_allocator.free(ends_array);

    var axes_array: ?[]i64 = null;
    if (axes) |axes_tensor| {
        axes_array = try tensorToI64Array(T1, axes_tensor);
    }
    defer if (axes_array) |arr| pkg_allocator.free(arr);

    var steps_array: ?[]i64 = null;
    if (steps) |steps_tensor| {
        steps_array = try tensorToI64Array(T1, steps_tensor);
    }
    defer if (steps_array) |arr| pkg_allocator.free(arr);

    // Calculate output shape first
    const output_shape = try get_slice_output_shape(input.shape, starts_array, ends_array, axes_array, steps_array);
    defer pkg_allocator.free(output_shape);

    std.log.debug("\n[DEBUG] Calculated output_shape: {any}", .{output_shape});

    // Create output tensor using input's allocator for consistency
    var output = try Tensor(T).fromShape(input.allocator, output_shape);
    errdefer output.deinit();

    std.log.debug("\n[DEBUG] Created output tensor: shape={any}, size={d}", .{ output.shape, output.size });

    // Initialize output tensor to zero to help debug (only for debugging)
    for (output.data) |*val| {
        val.* = std.mem.zeroes(T);
    }

    try slice_lean(T, T1, input, starts, ends, axes, steps, &output);
    return output;
}

/// Lean version of slice that operates on an existing output tensor
pub fn slice_lean(comptime T: type, comptime T1: type, input: *Tensor(T), starts: *Tensor(T1), ends: *Tensor(T1), axes: ?*Tensor(T1), steps: ?*Tensor(T1), output: *Tensor(T)) !void {
    // Convert tensor inputs to arrays for easier processing
    const starts_array = try tensorToI64Array(T1, starts);
    defer pkg_allocator.free(starts_array);

    const ends_array = try tensorToI64Array(T1, ends);
    defer pkg_allocator.free(ends_array);

    var axes_array: ?[]i64 = null;
    if (axes) |axes_tensor| {
        axes_array = try tensorToI64Array(T1, axes_tensor);
    }
    defer if (axes_array) |arr| pkg_allocator.free(arr);

    var steps_array: ?[]i64 = null;
    if (steps) |steps_tensor| {
        steps_array = try tensorToI64Array(T1, steps_tensor);
    }
    defer if (steps_array) |arr| pkg_allocator.free(arr);

    // Validate input lengths
    if (starts_array.len != ends_array.len) return TensorError.InvalidSliceIndices;
    if (axes_array) |a| {
        if (a.len != starts_array.len) return TensorError.InvalidSliceIndices;
    }
    if (steps_array) |s| {
        if (s.len != starts_array.len) return TensorError.InvalidSliceIndices;
    }

    // Additional validation
    if (input.size == 0) {
        std.log.warn("Warning: Input tensor is empty", .{});
        return;
    }

    if (output.size == 0) {
        std.log.warn("Warning: Output tensor is empty", .{});
        return;
    }

    // Create arrays to store the effective indices and steps for each dimension
    var effective_starts = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(effective_starts);
    var effective_ends = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(effective_ends);
    var effective_steps = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(effective_steps);

    // Initialize effective values as per ONNX spec:
    for (0..input.shape.len) |i| {
        effective_starts[i] = 0;
        effective_ends[i] = @intCast(input.shape[i]);
        effective_steps[i] = 1;
    }

    // Process each slice parameter according to ONNX spec
    for (starts_array, 0..) |start, i| {
        // Determine the axis (handle negative axes by adding rank)
        var axis = if (axes_array) |a| a[i] else @as(i64, @intCast(i));
        if (axis < 0) {
            axis += @as(i64, @intCast(input.shape.len));
        }

        if (axis < 0 or axis >= @as(i64, @intCast(input.shape.len))) {
            return TensorError.InvalidSliceIndices;
        }

        const axis_usize = @as(usize, @intCast(axis));
        const dim_size = @as(i64, @intCast(input.shape[axis_usize]));

        // Get the step for this axis
        const step = if (steps_array) |s| s[i] else 1;
        if (step == 0) return TensorError.InvalidSliceStep;
        effective_steps[axis_usize] = step;

        // Handle negative starts by adding dim_size
        var effective_start = if (start < 0) start + dim_size else start;

        // Handle negative ends by adding dim_size
        var effective_end = if (ends_array[i] < 0) ends_array[i] + dim_size else ends_array[i];

        // Clamp starts and ends according to ONNX spec
        if (step > 0) {
            effective_start = @max(0, @min(effective_start, dim_size));
            effective_end = @max(0, @min(effective_end, dim_size));
        } else {
            effective_start = @max(0, @min(effective_start, dim_size - 1));
            effective_end = @max(-1, @min(effective_end, dim_size - 1));
        }

        effective_starts[axis_usize] = effective_start;
        effective_ends[axis_usize] = effective_end;
    }

    // Now perform the actual slicing using numpy-like semantics
    const output_ndim = output.shape.len;
    const input_ndim = input.shape.len;

    std.log.debug("\n[DEBUG] Starting slice copy operation:", .{});
    std.log.debug("\n  input.shape: {any}", .{input.shape});
    std.log.debug("\n  output.shape: {any}", .{output.shape});
    std.log.debug("\n  output.size: {d}", .{output.size});
    std.log.debug("\n  effective_starts: {any}", .{effective_starts});
    std.log.debug("\n  effective_ends: {any}", .{effective_ends});
    std.log.debug("\n  effective_steps: {any}", .{effective_steps});

    var output_coords = try pkg_allocator.alloc(usize, output_ndim);
    defer pkg_allocator.free(output_coords);
    var input_coords = try pkg_allocator.alloc(usize, input_ndim);
    defer pkg_allocator.free(input_coords);

    // Iterate through each element in the output tensor
    for (0..output.size) |output_idx| {
        // Convert flat output index to multi-dimensional coordinates
        var temp_idx = output_idx;
        for (0..output_ndim) |i| {
            const dim_idx = output_ndim - 1 - i;
            output_coords[dim_idx] = temp_idx % output.shape[dim_idx];
            temp_idx /= output.shape[dim_idx];
        }

        // Map output coordinates to input coordinates using the slice parameters
        for (0..output_ndim) |dim| {
            const output_coord = @as(i64, @intCast(output_coords[dim]));
            const input_coord = effective_starts[dim] + output_coord * effective_steps[dim];

            if (input_coord < 0 or input_coord >= @as(i64, @intCast(input.shape[dim]))) {
                return TensorError.InvalidSliceIndices;
            }

            input_coords[dim] = @as(usize, @intCast(input_coord));
        }

        // Copy the data
        const input_idx = try input.flatten_index(input_coords);
        if (input_idx >= input.size) {
            return TensorError.InvalidSliceIndices;
        }

        output.data[output_idx] = input.data[input_idx];
    }
}

/// Helper function to convert a tensor to an i64 array
fn tensorToI64Array(comptime T: type, tensor: *Tensor(T)) ![]i64 {
    var result = try pkg_allocator.alloc(i64, tensor.size);
    errdefer pkg_allocator.free(result);

    for (0..tensor.size) |i| {
        result[i] = switch (T) {
            i8 => @as(i64, tensor.data[i]),
            i16 => @as(i64, tensor.data[i]),
            i32 => @as(i64, tensor.data[i]),
            i64 => tensor.data[i],
            u8 => @as(i64, @intCast(tensor.data[i])),
            u16 => @as(i64, @intCast(tensor.data[i])),
            u32 => @as(i64, @intCast(tensor.data[i])),
            u64 => @as(i64, @intCast(tensor.data[i])),
            else => {
                std.log.err("Unsupported tensor type for slice indices: {}", .{T});
                return TensorError.InvalidSliceIndices;
            },
        };
    }

    return result;
}
