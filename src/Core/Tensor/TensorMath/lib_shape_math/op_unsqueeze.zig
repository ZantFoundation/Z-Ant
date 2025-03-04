const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

/// Implements https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
/// Insert single-dimensional entries into the shape of the data tensor.
pub fn unsqueeze(comptime T: type, data: *Tensor(T), axes: *Tensor(i64)) !Tensor(T) {

    // Output rank
    const out_rank = data.shape.len + axes.size;
    const conv_out_rank: i64 = @intCast(out_rank);

    for (0..axes.data.len) |i| {

        // Check if axes are within bounds
        if (axes.data[i] < -conv_out_rank or axes.data[i] >= out_rank) {
            return TensorError.AxisOutOfBounds;
        }

        // Check for duplicates
        for (0..i) |j| {
            if (axes.data[i] == axes.data[j]) {
                return TensorError.DuplicateAxis;
            }
        }
    }

    // Create, fill and return output tensor
    var output = try Tensor(T).init(data.allocator);

    try unsqueeze_lean(T, data, axes, &output);

    return output;
}

/// Lean version of unsqueeze, note that previous information stored in output tensor is lost
pub fn unsqueeze_lean(comptime T: type, data: *Tensor(T), axes: *Tensor(i64), output: *Tensor(T)) !void {

    // Output rank
    const out_rank = data.shape.len + axes.size;

    // Convert negative axis
    var actual_axes = try data.allocator.alloc(usize, axes.size);
    defer data.allocator.free(actual_axes);

    for (0..axes.size) |i| {
        var conv: i64 = axes.data[i];
        if (conv < 0) {
            conv += @intCast(out_rank);
        }
        const new_axis: usize = @intCast(conv);
        actual_axes[i] = new_axis;
    }

    // Preparing the output shape
    var new_shape = try data.allocator.alloc(usize, out_rank);
    var is_unsqueezed = try data.allocator.alloc(bool, out_rank);
    defer data.allocator.free(new_shape);
    defer data.allocator.free(is_unsqueezed);

    // Initialize support array
    @memset(is_unsqueezed, false);

    // Adding new mono dimentions and setting support array.
    for (0..actual_axes.len) |i| {
        new_shape[actual_axes[i]] = 1;
        is_unsqueezed[actual_axes[i]] = true;
    }

    // Setting positions not marked using data shape
    var data_index: usize = 0;
    for (0..out_rank) |i| {
        if (!is_unsqueezed[i]) {
            new_shape[i] = data.shape[data_index];
            data_index += 1;
        }
    }

    // Modify output tensor
    try output.fill(data.data, new_shape);
}
