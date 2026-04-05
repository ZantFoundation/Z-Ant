const std = @import("std");
const zant = @import("zant");


const pkg_allocator = zant.utils.allocator.allocator;

/// Calculate the output shape of a slice operation without performing the slice
pub fn get_slice_output_shape(input_shape: []const usize, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) ![]usize {
    std.log.debug("\n[DEBUG] get_slice_output_shape input:", .{});
    std.log.debug("\n  input_shape: {any}", .{input_shape});
    std.log.debug("\n  starts: {any}", .{starts});
    std.log.debug("\n  ends: {any}", .{ends});
    std.log.debug("\n  axes: {any}", .{axes});
    std.log.debug("\n  steps: {any}", .{steps});

    // Validate input lengths
    if (starts.len != ends.len) return error.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return error.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return error.InvalidSliceIndices;
    }

    // Create arrays to store the effective indices and steps for each dimension
    var effective_starts = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(effective_starts);
    var effective_ends = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(effective_ends);
    var effective_steps = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(effective_steps);

    // Initialize effective values as per ONNX spec
    for (0..input_shape.len) |i| {
        effective_starts[i] = 0;
        effective_ends[i] = @intCast(input_shape[i]);
        effective_steps[i] = 1;
    }

    std.log.debug("\n[DEBUG] Initial effective values:", .{});
    std.log.debug("\n  effective_starts: {any}", .{effective_starts});
    std.log.debug("\n  effective_ends: {any}", .{effective_ends});
    std.log.debug("\n  effective_steps: {any}", .{effective_steps});

    // Process each slice parameter
    for (starts, 0..) |start, i| {
        // Determine the axis (handle negative axes by adding rank)
        var axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        if (axis < 0) {
            axis += @as(i64, @intCast(input_shape.len));
        }

        if (axis < 0 or axis >= @as(i64, @intCast(input_shape.len))) {
            return error.InvalidSliceIndices;
        }

        const axis_usize = @as(usize, @intCast(axis));
        const dim_size = @as(i64, @intCast(input_shape[axis_usize]));

        // Get the step for this axis
        const step = if (steps) |s| s[i] else 1;
        if (step == 0) return error.InvalidSliceStep;
        effective_steps[axis_usize] = step;

        // Handle negative starts by adding dim_size
        var effective_start = if (start < 0) start + dim_size else start;

        // Handle negative ends by adding dim_size
        var effective_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];

        // Clamp starts and ends according to ONNX spec
        if (step > 0) {
            // Positive stepping: clamp starts to [0, dim_size] and ends to [0, dim_size]
            effective_start = @max(0, @min(effective_start, dim_size));
            effective_end = @max(0, @min(effective_end, dim_size));
        } else {
            // Negative stepping: clamp starts to [0, dim_size-1] and ends to [-1, dim_size-1]
            effective_start = @max(0, @min(effective_start, dim_size - 1));
            effective_end = @max(-1, @min(effective_end, dim_size - 1));
        }

        effective_starts[axis_usize] = effective_start;
        effective_ends[axis_usize] = effective_end;

        std.log.debug("\n[DEBUG] After processing axis {d} (original axis {d}):", .{ axis_usize, if (axes) |a| a[i] else @as(i64, @intCast(i)) });
        std.log.debug("\n  dim_size: {d}", .{dim_size});
        std.log.debug("\n  step: {d}", .{step});
        std.log.debug("\n  effective_start: {d}", .{effective_start});
        std.log.debug("\n  effective_end: {d}", .{effective_end});
    }

    std.log.debug("\n[DEBUG] Final effective values before shape calculation:", .{});
    std.log.debug("\n  effective_starts: {any}", .{effective_starts});
    std.log.debug("\n  effective_ends: {any}", .{effective_ends});
    std.log.debug("\n  effective_steps: {any}", .{effective_steps});

    // Calculate output shape for each dimension using the numpy slicing formula
    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    for (0..input_shape.len) |i| {
        const start = effective_starts[i];
        const end = effective_ends[i];
        const step = effective_steps[i];

        var dim_size: usize = 0;

        if (step > 0) {
            // Forward iteration: similar to numpy's range(start, end, step)
            if (end > start) {
                // Calculate number of elements: ceiling division of (end - start) / step
                const range = end - start;
                dim_size = @as(usize, @intCast(@divTrunc(range + step - 1, step)));
                std.log.debug("\n[DEBUG] Positive step calculation for dim {d}:", .{i});
                std.log.debug("\n  range = end({d}) - start({d}) = {d}", .{ end, start, range });
                std.log.debug("\n  dim_size = ceil(range / step) = ceil({d} / {d}) = {d}", .{ range, step, dim_size });
            }
        } else {
            // Backward iteration: similar to numpy's range(start, end, step) where step < 0
            if (start > end) {
                // For negative steps, we go from start down to (but not including) end
                // Similar to range(start, end, step) in Python where step < 0
                const range = start - end;
                const abs_step = -step;
                dim_size = @as(usize, @intCast(@divTrunc(range + abs_step - 1, abs_step)));
                std.log.debug("\n[DEBUG] Negative step calculation for dim {d}:", .{i});
                std.log.debug("\n  range = start({d}) - end({d}) = {d}", .{ start, end, range });
                std.log.debug("\n  abs_step = {d}", .{abs_step});
                std.log.debug("\n  dim_size = ceil(range / abs_step) = ceil({d} / {d}) = {d}", .{ range, abs_step, dim_size });
            }
        }

        output_shape[i] = dim_size;
        std.log.warn("\n[SLICE DEBUG] Dim {d}: start={d}, end={d}, step={d} -> size={d}", .{ i, start, end, step, dim_size });
    }

    std.log.warn("\n[SLICE DEBUG] Final computed output_shape: {any}\n", .{output_shape});
    return output_shape;
}
