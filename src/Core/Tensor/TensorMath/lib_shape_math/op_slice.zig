const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
/// Takes a tensor and extracts a slice along multiple axes.
/// starts: Starting indices for each axis
/// ends: Ending indices for each axis (exclusive)
/// axes: Which axes to slice (if null, assumes [0,1,2,...])
/// steps: Step sizes for each axis (if null, assumes all 1s)
pub fn slice_onnx(comptime T: type, input: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
    // Create output tensor using input's allocator for consistency
    var output = try Tensor(T).fromShape(input.allocator, input.shape);
    errdefer output.deinit();

    try lean_slice_onnx(T, input, starts, ends, axes, steps, &output);
    return output;
}

/// Lean version of slice_onnx that operates on an existing output tensor
pub fn lean_slice_onnx(comptime T: type, input: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64, output: *Tensor(T)) !void {
    // Validate input lengths
    if (starts.len != ends.len) return TensorError.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return TensorError.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return TensorError.InvalidSliceIndices;
    }

    // Create arrays to store the actual indices and steps for each dimension
    var actual_starts = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_starts);
    var actual_ends = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_ends);
    var actual_steps = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_steps);

    // Initialize with defaults (full range, step 1)
    for (0..input.shape.len) |i| {
        actual_starts[i] = 0;
        actual_ends[i] = @intCast(input.shape[i]);
        actual_steps[i] = 1;
    }

    // Update with provided values
    for (starts, 0..) |start, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        const axis_usize = if (axis < 0) @as(usize, @intCast(axis + @as(i64, @intCast(input.shape.len)))) else @as(usize, @intCast(axis));
        if (axis_usize >= input.shape.len) return TensorError.InvalidSliceIndices;

        const dim_size = @as(i64, @intCast(input.shape[axis_usize]));

        // Handle negative indices and clamp to valid range
        var actual_start = if (start < 0) start + dim_size else start;
        actual_start = @max(0, @min(actual_start, dim_size));
        actual_starts[axis_usize] = actual_start;

        var actual_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];
        if (steps) |s| {
            if (s[i] < 0) {
                // For negative steps, if end is negative, we want to include 0
                actual_end = if (ends[i] < 0) -1 else actual_end;
            } else {
                actual_end = @max(0, @min(actual_end, dim_size));
            }
        } else {
            actual_end = @max(0, @min(actual_end, dim_size));
        }
        actual_ends[axis_usize] = actual_end;

        if (steps) |s| {
            if (s[i] == 0) return TensorError.InvalidSliceStep;
            actual_steps[axis_usize] = s[i];
        }
    }

    // Calculate output shape
    var total_elements: usize = 1;
    for (0..input.shape.len) |i| {
        const start = actual_starts[i];
        const end = actual_ends[i];
        const step = actual_steps[i];

        var dim_size: usize = 0;
        if (step > 0) {
            if (end > start) {
                dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
            }
        } else {
            if (start > end) {
                // For negative steps, we need to handle the range differently
                // Add 1 to end because end is exclusive
                const range = start - (end + 1);
                const abs_step = -step;
                dim_size = @intCast(@divTrunc(range + abs_step - 1, abs_step));
            }
        }
        output.shape[i] = dim_size;
        total_elements *= dim_size;
    }

    // Resize output data if needed
    if (output.data.len != total_elements) {
        // Don't free the memory directly as it might have been allocated by a different allocator
        // Instead, deinitialize and reinitialize with the correct size
        if (output.data.len > 0) {
            // Just abandon the old data to avoid potential invalid free
            // This may leak memory but is better than crashing
        }
        output.data = try output.allocator.alloc(T, total_elements);
    }
    output.size = total_elements;

    // Helper function to convert flat index to coordinates
    var input_coords = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(input_coords);
    var output_coords = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(output_coords);

    // Copy data
    var output_idx: usize = 0;
    while (output_idx < total_elements) : (output_idx += 1) {
        // Convert output_idx to coordinates
        var temp = output_idx;
        for (0..input.shape.len) |i| {
            const dim_i = input.shape.len - 1 - i;
            output_coords[dim_i] = temp % output.shape[dim_i];
            temp /= output.shape[dim_i];
        }

        // Calculate input coordinates
        for (0..input.shape.len) |i| {
            const coord = @as(i64, @intCast(output_coords[i]));
            input_coords[i] = @intCast(actual_starts[i] + coord * actual_steps[i]);
        }

        // Get input value
        const input_idx = try input.flatten_index(input_coords);
        output.data[output_idx] = input.data[input_idx];
    }
}

/// Calculate the output shape of a slice operation without performing the slice
pub fn get_slice_output_shape(input_shape: []const usize, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) ![]usize {
    std.debug.print("\n[DEBUG] get_slice_output_shape input:", .{});
    std.debug.print("\n  input_shape: {any}", .{input_shape});
    std.debug.print("\n  starts: {any}", .{starts});
    std.debug.print("\n  ends: {any}", .{ends});
    std.debug.print("\n  axes: {any}", .{axes});
    std.debug.print("\n  steps: {any}", .{steps});

    // Validate input lengths
    if (starts.len != ends.len) return TensorError.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return TensorError.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return TensorError.InvalidSliceIndices;
    }

    // Create arrays to store the actual indices and steps for each dimension
    var actual_starts = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_starts);
    var actual_ends = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_ends);
    var actual_steps = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_steps);

    // Initialize with defaults (full range, step 1)
    for (0..input_shape.len) |i| {
        actual_starts[i] = 0;
        actual_ends[i] = @intCast(input_shape[i]);
        actual_steps[i] = 1;
    }

    std.debug.print("\n[DEBUG] Initial values:", .{});
    std.debug.print("\n  actual_starts: {any}", .{actual_starts});
    std.debug.print("\n  actual_ends: {any}", .{actual_ends});
    std.debug.print("\n  actual_steps: {any}", .{actual_steps});

    // Update with provided values
    for (starts, 0..) |start, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        const axis_usize = if (axis < 0) @as(usize, @intCast(axis + @as(i64, @intCast(input_shape.len)))) else @as(usize, @intCast(axis));
        if (axis_usize >= input_shape.len) return TensorError.InvalidSliceIndices;

        const dim_size = @as(i64, @intCast(input_shape[axis_usize]));

        // Handle negative indices and clamp to valid range
        var actual_start = if (start < 0) start + dim_size else start;
        actual_start = @max(0, @min(actual_start, dim_size));
        actual_starts[axis_usize] = actual_start;

        var actual_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];
        if (steps) |s| {
            if (s[i] < 0) {
                // For negative steps, if end is negative, we want to include 0
                actual_end = if (ends[i] < 0) -1 else actual_end;
            } else {
                actual_end = @max(0, @min(actual_end, dim_size));
            }
        } else {
            actual_end = @max(0, @min(actual_end, dim_size));
        }
        actual_ends[axis_usize] = actual_end;

        if (steps) |s| {
            if (s[i] == 0) return TensorError.InvalidSliceStep;
            actual_steps[axis_usize] = s[i];
        }

        std.debug.print("\n[DEBUG] After processing axis {d}:", .{axis});
        std.debug.print("\n  dim_size: {d}", .{dim_size});
        std.debug.print("\n  actual_start: {d}", .{actual_start});
        std.debug.print("\n  actual_end: {d}", .{actual_end});
        std.debug.print("\n  actual_step: {d}", .{actual_steps[axis_usize]});
    }

    std.debug.print("\n[DEBUG] Final values before shape calculation:", .{});
    std.debug.print("\n  actual_starts: {any}", .{actual_starts});
    std.debug.print("\n  actual_ends: {any}", .{actual_ends});
    std.debug.print("\n  actual_steps: {any}", .{actual_steps});

    // Calculate output shape
    // Calculate output shape
    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    for (0..input_shape.len) |i| {
        const start = actual_starts[i];
        const end = actual_ends[i];
        const step = actual_steps[i];

        var dim_size: usize = 0;
        if (step > 0) {
            if (end > start) {
                dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
                std.debug.print("\n[DEBUG] Positive step calculation for dim {d}:", .{i});
                std.debug.print("\n  end ({d}) - start ({d}) = {d}", .{ end, start, end - start });
                std.debug.print("\n  (end-start) + step({d}) - 1 = {d}", .{ step, (end - start) + step - 1 });
                std.debug.print("\n  final dim_size = {d}", .{dim_size});
            }
        } else {
            if (start > end) {
                // For negative steps, treat end as inclusive.
                const range = start - end;
                dim_size = @intCast((@divTrunc(range, -step)) + 1);
                std.debug.print("\n[DEBUG] Negative step calculation for dim {d}:", .{i});
                std.debug.print("\n  start ({d}) - end ({d}) = range ({d})", .{ start, end, range });
                std.debug.print("\n  (range) / abs(step) + 1 = {d}", .{dim_size});
            }
        }
        output_shape[i] = dim_size;
    }

    std.debug.print("\n[DEBUG] Final output_shape: {any}\n", .{output_shape});
    return output_shape;
}
