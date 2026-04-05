const std = @import("std");
const zant = @import("../../../../zant.zig");


const pkg_allocator = zant.utils.allocator.allocator;

/// Calculate the output shape for an ONNX Unsqueeze operation without performing the operation
pub fn get_unsqueeze_output_shape(input_shape: []const usize, axes: []const i64) ![]usize {
    // Output rank
    const out_rank = input_shape.len + axes.len;

    // Convert negative axes to positive
    var actual_axes = try pkg_allocator.alloc(usize, axes.len);
    defer pkg_allocator.free(actual_axes);

    for (axes, 0..) |axis, i| {
        var conv: i64 = axis;
        if (conv < 0) {
            conv += @intCast(out_rank);
        }
        if (conv < 0 or conv >= out_rank) {
            return error.AxisOutOfBounds;
        }
        const new_axis: usize = @intCast(conv);

        // Check for duplicates
        for (0..i) |j| {
            if (actual_axes[j] == new_axis) {
                return error.DuplicateAxis;
            }
        }
        actual_axes[i] = new_axis;
    }

    // Create output shape array
    var output_shape = try pkg_allocator.alloc(usize, out_rank);

    // Create and initialize support array to track unsqueezed dimensions
    var is_unsqueezed = try pkg_allocator.alloc(bool, out_rank);
    defer pkg_allocator.free(is_unsqueezed);
    @memset(is_unsqueezed, false);

    // Mark unsqueezed dimensions and set them to 1
    for (actual_axes) |axis| {
        output_shape[axis] = 1;
        is_unsqueezed[axis] = true;
    }

    // Fill remaining dimensions with input shape values
    var input_idx: usize = 0;
    for (0..out_rank) |i| {
        if (!is_unsqueezed[i]) {
            output_shape[i] = input_shape[input_idx];
            input_idx += 1;
        }
    }

    return output_shape;
}
