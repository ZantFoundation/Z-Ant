const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const TensorError = @import("errorHandler").TensorError;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const PoolingType = @import("layer").poolingLayer.PoolingType;

/// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
/// Takes a tensor and extracts a slice along multiple axes.
/// starts: Starting indices for each axis
/// ends: Ending indices for each axis (exclusive)
/// axes: Which axes to slice (if null, assumes [0,1,2,...])
/// steps: Step sizes for each axis (if null, assumes all 1s)
pub fn slice(comptime T: anytype, data: *const Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
    // Validate input lengths
    if (starts.len != ends.len) return TensorMathError.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return TensorMathError.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return TensorMathError.InvalidSliceIndices;
    }

    for (starts, 0..) |_, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        
        const axis_usize = if (axis < 0)
            @as(usize, @intCast(axis + @as(i64, @intCast(data.shape.len))))
        else
            @as(usize, @intCast(axis));

        if (axis_usize >= data.shape.len) return TensorMathError.InvalidSliceIndices;

        if (steps) |s| {
            if (s[i] == 0) return TensorMathError.InvalidSliceStep;
        }
    }

    // Create output tensor
    var output = try Tensor(T).init(data.allocator);
    errdefer output.deinit();

    // Call lean_slice
    try lean_slice(T, data, starts, ends, axes, steps, &output);

    return output;
}

pub fn lean_slice(comptime T: anytype, data: *const Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64, output: *Tensor(T)) !void {
    // Create arrays to store the actual indices and steps for each dimension
    var actual_starts = try data.allocator.alloc(i64, data.shape.len);
    defer data.allocator.free(actual_starts);
    var actual_ends = try data.allocator.alloc(i64, data.shape.len);
    defer data.allocator.free(actual_ends);
    var actual_steps = try data.allocator.alloc(i64, data.shape.len);
    defer data.allocator.free(actual_steps);

    // Initialize with defaults (full range, step 1)
    for (0..data.shape.len) |i| {
        actual_starts[i] = 0;
        actual_ends[i] = @intCast(data.shape[i]);
        actual_steps[i] = 1;
    }

    // Update with provided values
    for (starts, 0..) |start, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));

        const axis_usize = if (axis < 0)
            @as(usize, @intCast(axis + @as(i64, @intCast(data.shape.len))))
        else
            @as(usize, @intCast(axis));

        const dim_size = @as(i64, @intCast(data.shape[axis_usize]));

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
            actual_steps[axis_usize] = s[i];
        }
    }

    // Calculate output shape
    var output_shape = try data.allocator.alloc(usize, data.shape.len);
    errdefer data.allocator.free(output_shape);

    var total_elements: usize = 1;
    for (0..data.shape.len) |i| {
        const start = actual_starts[i];
        const end = actual_ends[i];
        const step = actual_steps[i];

        var dim_size: usize = 0;
        if (step > 0) {
            if (end > start) {
                dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
                // std.debug.print("\nPositive step: start={}, end={}, step={}, dim_size={}", .{ start, end, step, dim_size });
            }
        } else {
            if (start > end) {
                // For negative steps, we need to handle the range differently
                // Add 1 to end because end is exclusive
                const range = start - (end + 1);
                const abs_step = -step;
                dim_size = @intCast(@divTrunc(range + abs_step - 1, abs_step));
                // std.debug.print("\nNegative step: start={}, end={}, step={}, range={}, abs_step={}, dim_size={}", .{ start, end, step, range, abs_step, dim_size });
            }
        }
        // std.debug.print("\nDimension {}: dim_size={}", .{ i, dim_size });
        output_shape[i] = dim_size;
        total_elements *= dim_size;
    }

    // Allocate output data
    var output_data = try data.allocator.alloc(T, total_elements);
    errdefer data.allocator.free(output_data);

    // Helper function to convert flat index to coordinates
    var input_coords = try data.allocator.alloc(usize, data.shape.len);
    defer data.allocator.free(input_coords);
    var output_coords = try data.allocator.alloc(usize, data.shape.len);
    defer data.allocator.free(output_coords);

    // Copy data
    var output_idx: usize = 0;
    std.debug.print("\nTotal elements: {}", .{total_elements});
    while (output_idx < total_elements) : (output_idx += 1) {
        // Convert output_idx to coordinates
        var temp = output_idx;
        for (0..data.shape.len) |i| {
            const dim_i = data.shape.len - 1 - i;
            output_coords[dim_i] = temp % output_shape[dim_i];
            temp /= output_shape[dim_i];
        }

        // Calculate input coordinates
        for (0..data.shape.len) |i| {
            const coord = @as(i64, @intCast(output_coords[i]));
            input_coords[i] = @intCast(actual_starts[i] + coord * actual_steps[i]);
            // std.debug.print("\noutput_coord[{}]={}, input_coord[{}]={}", .{ i, output_coords[i], i, input_coords[i] });
        }

        // Get input value
        const input_idx = try data.flatten_index(input_coords);
        output_data[output_idx] = data.data[input_idx];
        // std.debug.print("\noutput_idx={}, input_idx={}, value={}", .{ output_idx, input_idx, output_data[output_idx] });
    }

    output.data = output_data;
    output.shape = output_shape;
    output.size = total_elements;
    output.allocator = data.allocator;
}
