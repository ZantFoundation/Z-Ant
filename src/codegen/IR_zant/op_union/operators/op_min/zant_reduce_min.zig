const IR_zant = @import("IR_zant");
const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// Reduces a tensor along specified axes to find minimum values.
pub fn reduce_min(comptime T: anytype, input: *Tensor(T), axes: ?[]const i64, keepdims: bool) !Tensor(T) {
    if (axes == null) {
        // Reduce over all dimensions - find global minimum
        var output_shape: []usize = undefined;
        if (keepdims) {
            output_shape = try pkg_allocator.alloc(usize, input.shape.len);
            @memset(output_shape, 1);
        } else {
            output_shape = try pkg_allocator.alloc(usize, 0);
        }

        var output = try Tensor(T).fromShape(input.allocator, output_shape);
        pkg_allocator.free(output_shape);
        errdefer output.deinit();

        if (input.size == 0) return error.InvalidDimensions;

        var min_val = input.data[0];
        for (input.data[1..]) |val| {
            if (val < min_val) min_val = val;
        }

        if (output.size > 0) output.data[0] = min_val;
        return output;
    }

    // For specific axes, we'd need more complex reduction logic
    // For now, return a copy (placeholder implementation)
    return input.copy();
}

/// Lean version of reduce_min that operates on pre-allocated output tensor.
pub fn reduce_min_lean(comptime T: anytype, input: *Tensor(T), output: *Tensor(T), axes: ?[]const i64, _: bool) !void {
    if (axes == null) {
        // Global reduction
        if (input.size == 0) return error.InvalidDimensions;

        var min_val = input.data[0];
        for (input.data[1..]) |val| {
            if (val < min_val) min_val = val;
        }

        if (output.size > 0) output.data[0] = min_val;
        return;
    }

    // For specific axes, placeholder implementation
    if (input.size == output.size) {
        @memcpy(output.data, input.data);
    }
}
