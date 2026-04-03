const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- TOPK OPERATOR ---------------------

/// Element structure for TopK sorting
const TopKElement = struct {
    value: f32,
    index: usize,
};

/// Computes the output shape for the TopK operator.
/// Returns two shapes: one for values and one for indices.
pub fn get_topk_output_shape(
    input_shape: []const usize,
    k: usize,
    axis: i64,
) !struct { values_shape: []usize, indices_shape: []usize } {
    if (input_shape.len == 0) {
        return TensorMathError.InvalidInput;
    }

    // Normalize axis
    const normalized_axis = if (axis < 0)
        @as(usize, @intCast(@as(i64, @intCast(input_shape.len)) + axis))
    else
        @as(usize, @intCast(axis));

    if (normalized_axis >= input_shape.len) {
        return TensorMathError.InvalidInput;
    }

    // Check if k is valid
    if (k > input_shape[normalized_axis]) {
        return TensorMathError.InvalidInput;
    }

    // Create output shapes
    const values_shape = try pkg_allocator.alloc(usize, input_shape.len);
    const indices_shape = try pkg_allocator.alloc(usize, input_shape.len);

    @memcpy(values_shape, input_shape);
    @memcpy(indices_shape, input_shape);

    values_shape[normalized_axis] = k;
    indices_shape[normalized_axis] = k;

    return .{ .values_shape = values_shape, .indices_shape = indices_shape };
}

/// Applies TopK operation, allocating new output tensors.
pub fn topk(
    comptime T: type,
    input: *const Tensor(T),
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
) !struct { values: Tensor(T), indices: Tensor(i64) } {
    // Validate input
    if (input.data.len == 0) {
        return TensorError.ZeroSizeTensor;
    }

    // Compute output shapes
    const output_shapes = try get_topk_output_shape(input.shape, k, axis);
    defer pkg_allocator.free(output_shapes.values_shape);
    defer pkg_allocator.free(output_shapes.indices_shape);

    // Allocate output tensors
    var values = try Tensor(T).fromShape(&pkg_allocator, output_shapes.values_shape);
    errdefer values.deinit();

    var indices = try Tensor(i64).fromShape(&pkg_allocator, output_shapes.indices_shape);
    errdefer indices.deinit();

    try topk_lean(T, input, &values, &indices, k, axis, largest, sorted);

    return .{ .values = values, .indices = indices };
}

/// Applies TopK operation on pre-allocated output tensors.
pub fn topk_lean(
    comptime T: type,
    input: *const Tensor(T),
    values_output: *Tensor(T),
    indices_output: *Tensor(i64),
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
) !void {
    std.log.warn("\n[TOPK DEBUG] === Starting TopK operation ===", .{});
    std.log.warn("[TOPK DEBUG] Input shape: {any}, size: {d}", .{ input.shape, input.size });
    std.log.warn("[TOPK DEBUG] Values output shape: {any}, size: {d}", .{ values_output.shape, values_output.size });
    std.log.warn("[TOPK DEBUG] Indices output shape: {any}, size: {d}", .{ indices_output.shape, indices_output.size });
    std.log.warn("[TOPK DEBUG] k: {d}, axis: {d}, largest: {}, sorted: {}", .{ k, axis, largest, sorted });
    // Normalize axis
    const normalized_axis = if (axis < 0)
        @as(usize, @intCast(@as(i64, @intCast(input.shape.len)) + axis))
    else
        @as(usize, @intCast(axis));

    if (normalized_axis >= input.shape.len) {
        return TensorMathError.InvalidInput;
    }

    const axis_size = input.shape[normalized_axis];
    if (k > axis_size) {
        return TensorMathError.InvalidInput;
    }

    // Calculate strides
    const strides = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(strides);

    strides[input.shape.len - 1] = 1;
    if (input.shape.len > 1) {
        var i = input.shape.len - 2;
        while (true) {
            strides[i] = strides[i + 1] * input.shape[i + 1];
            if (i == 0) break;
            i -= 1;
        }
    }

    // Calculate outer and inner sizes
    var outer_size: usize = 1;
    for (0..normalized_axis) |i| {
        outer_size *= input.shape[i];
    }

    var inner_size: usize = 1;
    for ((normalized_axis + 1)..input.shape.len) |i| {
        inner_size *= input.shape[i];
    }

    // Process each outer×inner combination
    for (0..outer_size) |outer_idx| {
        for (0..inner_size) |inner_idx| {
            // Collect elements along the axis
            var elements = try pkg_allocator.alloc(TopKElement, axis_size);
            defer pkg_allocator.free(elements);

            for (0..axis_size) |axis_idx| {
                const linear_idx = calculateLinearIndex(outer_idx, axis_idx, inner_idx, normalized_axis, strides);
                elements[axis_idx] = TopKElement{
                    .value = @as(f32, @floatCast(input.data[linear_idx])),
                    .index = axis_idx,
                };
            }

            // Sort elements based on largest flag
            if (largest) {
                std.sort.block(TopKElement, elements, {}, struct {
                    fn lessThan(_: void, lhs: TopKElement, rhs: TopKElement) bool {
                        if (lhs.value != rhs.value) {
                            return lhs.value > rhs.value; // Descending order for largest
                        }
                        return lhs.index < rhs.index; // Stable sort by index
                    }
                }.lessThan);
            } else {
                std.sort.block(TopKElement, elements, {}, struct {
                    fn lessThan(_: void, lhs: TopKElement, rhs: TopKElement) bool {
                        if (lhs.value != rhs.value) {
                            return lhs.value < rhs.value; // Ascending order for smallest
                        }
                        return lhs.index < rhs.index; // Stable sort by index
                    }
                }.lessThan);
            }

            // If not sorted, we might need to re-sort by original indices
            if (!sorted) {
                // Extract top-k elements first
                const topk_elements = try pkg_allocator.alloc(TopKElement, k);
                defer pkg_allocator.free(topk_elements);

                @memcpy(topk_elements, elements[0..k]);

                // Sort by original indices to maintain input order
                std.sort.block(TopKElement, topk_elements, {}, struct {
                    fn lessThan(_: void, lhs: TopKElement, rhs: TopKElement) bool {
                        return lhs.index < rhs.index;
                    }
                }.lessThan);

                // Copy back the results
                @memcpy(elements[0..k], topk_elements);
            }

            // Write results to output tensors
            for (0..k) |i| {
                const output_idx = calculateLinearIndex(outer_idx, i, inner_idx, normalized_axis, strides);
                std.log.warn("[TOPK DEBUG] Writing result {d}: output_idx={d}, values_output.data.len={d}, k={d}", .{ i, output_idx, values_output.data.len, k });
                if (output_idx >= values_output.data.len) {
                    std.log.err("[TOPK DEBUG] ERROR: output_idx ({d}) >= values_output.data.len ({d})", .{ output_idx, values_output.data.len });
                    return;
                }
                values_output.data[output_idx] = @as(T, @floatCast(elements[i].value));
                indices_output.data[output_idx] = @as(i64, @intCast(elements[i].index));
            }
        }
    }
}

/// Calculate linear memory index from multi-dimensional indices using strides
inline fn calculateLinearIndex(
    outer_idx: usize,
    axis_idx: usize,
    inner_idx: usize,
    axis: usize,
    strides: []const usize,
) usize {
    var linear_idx: usize = 0;

    // Add contribution from outer dimensions
    var remaining_outer = outer_idx;
    for (0..axis) |dim| {
        const dim_idx = remaining_outer / strides[dim + 1];
        remaining_outer = remaining_outer % strides[dim + 1];
        linear_idx += dim_idx * strides[dim];
    }

    // Add contribution from axis dimension
    linear_idx += axis_idx * strides[axis];

    // Add contribution from inner dimensions
    var remaining_inner = inner_idx;
    for ((axis + 1)..strides.len) |dim| {
        const dim_stride = if (dim + 1 < strides.len) strides[dim + 1] else 1;
        const dim_idx = remaining_inner / dim_stride;
        remaining_inner = remaining_inner % dim_stride;
        linear_idx += dim_idx * strides[dim];
    }

    return linear_idx;
}
