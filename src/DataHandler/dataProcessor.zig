const std = @import("std");
const zant = @import("../zant.zig");
const Tensor = zant.core.tensor.Tensor;

pub const NormalizationType = enum {
    null,
    UnityBasedNormalizartion,
    StandardDeviationNormalization,
};

pub fn normalize(comptime T: anytype, tensor: *Tensor(T), normalizationType: NormalizationType) !void {
    switch (normalizationType) {
        NormalizationType.UnityBasedNormalizartion => try multidimNormalizeUnityBased(T, tensor),
        NormalizationType.StandardDeviationNormalization => try multidimNormalizeStandard(T, tensor),
        else => try multidimNormalizeUnityBased(T, tensor),
    }
}

/// Normalize each row in a multidimensional tensor
fn multidimNormalizeUnityBased(comptime T: anytype, tensor: *Tensor(T)) !void {
    // --- Type Checks ---
    // Ensure that T is a floating-point type
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    // --- Shape Checks ---
    // Expecting a tensor with at least 2 dimensions
    if (tensor.shape.len < 2) {
        return error.InvalidTensorShape;
    }

    // Determine normalization parameters based on tensor dimensionality
    var slice_size: usize = 1;
    var num_slices: usize = 1;

    if (tensor.shape.len == 2) {
        // For 2D tensors: [batch, features]
        slice_size = tensor.shape[1];
        num_slices = tensor.shape[0];
    } else if (tensor.shape.len == 4) {
        // For 4D tensors: [batch, channels, width, height]
        slice_size = tensor.shape[2] * tensor.shape[3];
        num_slices = tensor.shape[0] * tensor.shape[1];
    } else {
        // Unsupported tensor shape
        return error.UnsupportedTensorShape;
    }

    // Iterate over each slice and perform min-max normalization
    for (0..num_slices) |slice_idx| {
        // Calculate the starting index for the current slice in the flat data array
        const slice_offset = slice_idx * slice_size;

        // --- Step 1: Find Min and Max ---
        var slice_min: T = tensor.data[slice_offset];
        var slice_max: T = tensor.data[slice_offset];

        for (0..slice_size) |i| {
            const val = tensor.data[slice_offset + i];
            if (val > slice_max) slice_max = val;
            if (val < slice_min) slice_min = val;
        }

        // --- Step 2: Compute Delta ---
        const delta = slice_max - slice_min;

        // --- Step 3: Normalize Each Element ---
        for (0..slice_size) |i| {
            const idx = slice_offset + i;
            if (delta != 0) {
                tensor.data[idx] = (tensor.data[idx] - slice_min) / delta;
            } else {
                // If delta is zero (all elements are the same), set normalized value to 0
                tensor.data[idx] = 0;
            }
        }
    }
}

fn multidimNormalizeStandard(comptime T: anytype, tensor: *Tensor(T)) !void {
    std.debug.print("\n     standard normalize ", .{});
    // --- Type Checks ---
    // Ensure that T is a floating-point type
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    // --- Shape Checks ---
    // Expecting a tensor with at least 2 dimensions
    if (tensor.shape.len < 2) {
        return error.InvalidTensorShape;
    }

    const last_dim_size = tensor.shape[tensor.shape.len - 1];
    var num_slices: usize = 1;
    for (tensor.shape[0 .. tensor.shape.len - 1]) |dim| {
        num_slices *= dim;
    }

    // Small epsilon value to prevent division by zero
    const epsilon = @as(T, 1e-7);

    // Iterate over each slice
    for (0..num_slices) |slice| {
        const slice_offset = slice * last_dim_size;

        // --- Step 1: Compute Mean ---
        var sum: T = 0.0;
        for (0..last_dim_size) |i| {
            sum += tensor.data[slice_offset + i];
        }
        const mean = sum / @as(T, @floatFromInt(last_dim_size));

        // --- Step 2: Compute Variance ---
        var sum_sq_diff: T = 0.0;
        for (0..last_dim_size) |i| {
            const diff: T = tensor.data[slice_offset + i] - mean;
            sum_sq_diff += diff * diff;
        }
        const variance = sum_sq_diff / @as(T, @floatFromInt(last_dim_size));

        // --- Step 3: Compute Standard Deviation ---
        const std_dev = std.math.sqrt(variance + epsilon);

        // --- Step 4: Normalize Each Element ---
        for (0..last_dim_size) |i| {
            if (std_dev > 0) {
                tensor.data[slice_offset + i] = (tensor.data[slice_offset + i] - mean) / std_dev;
            } else {
                // If standard deviation is zero, set normalized value to 0
                tensor.data[slice_offset + i] = 0;
            }
        }
    }
}
