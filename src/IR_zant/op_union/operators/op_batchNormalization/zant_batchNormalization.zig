const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Onnx standard:
// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
pub fn batchNormalization(
    comptime T: anytype,
    comptime T1: anytype,
    comptime T2: anytype,
    input: *Tensor(T),
    scales: *Tensor(T1),
    B: *Tensor(T1),
    input_mean: *Tensor(T2),
    input_var: *Tensor(T2),
    epsilon: f32,
    momentum: f32,
    training_mode: bool,
) !Tensor(T) {

    //checks on the shapes
    if (input.size % scales.size != 0) return error.SizesDontMatch;

    const output_shape = try get_batchNormalization_output_shape(input.shape);
    var output = Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deint();

    try batchNormalization_lean(T, T1, T2, input, scales, B, input_mean, input_var, epsilon, momentum, training_mode, &output);
}

pub inline fn batchNormalization_lean(
    comptime T: anytype,
    comptime T1: anytype,
    comptime T2: anytype,
    input: *Tensor(T), // X
    scales: *Tensor(T1), //tensor of shape ©.
    B: *Tensor(T1), // tensor of shape ©.
    input_mean: *Tensor(T2), // tensor of shape ©.
    input_var: *Tensor(T2), // tensor of shape ©.
    epsilon: f32,
    momentum: f32,
    training_mode: bool,
    output: *Tensor(T), // Y
) !void {
    _ = momentum; //reduntant, used only for training
    if (training_mode) {
        std.log.warn("\n\nERROR: training_mode not available for batchNormalization!! \n", .{});
        return error.training_mode_NotAvailable;
    }

    const dims = input.shape.len;
    if (dims < 2) return error.InvalidDimensions; // Must have at least N, C dims

    // Assume Channel dimension is axis 1 (common for NCHW)
    const C = input.shape[1];
    if (C != scales.size or C != B.size or C != input_mean.size or C != input_var.size) {
        std.log.warn("ERROR: Channel size mismatch. Input C={}, Scale={}, B={}, Mean={}, Var={}\n", .{ C, scales.size, B.size, input_mean.size, input_var.size });
        return error.ChannelSizeMismatch;
    }
    if (output.shape.len != dims or !std.mem.eql(usize, output.shape, input.shape)) {
        std.log.warn("ERROR: Output shape mismatch. Input={any}, Output={any}\n", .{ input.shape, output.shape });
        return error.ShapeMismatch; // Ensure output shape matches input shape
    }

    // Precompute strides for faster index calculation (assuming row-major layout)
    var strides = try pkg_allocator.alloc(usize, dims);
    defer pkg_allocator.free(strides);
    strides[dims - 1] = 1;
    var d = dims - 1;
    while (d > 0) {
        d -= 1;
        strides[d] = strides[d + 1] * input.shape[d + 1];
    }

    // Iterate using multi-dimensional index
    var current_index = try pkg_allocator.alloc(usize, dims);
    defer pkg_allocator.free(current_index);
    // Zero initialize the index array manually instead of using @memset
    for (0..dims) |i| {
        current_index[i] = 0;
    }

    var flat_index: usize = 0;
    outer_loop: while (true) {
        const channel_index = current_index[1]; // Get the channel for this element (Axis 1)

        // --- Perform Calculation ---
        // Direct calculation assuming T, T1, T2 are compatible float types
        const mean = input_mean.data[channel_index];
        const variance = input_var.data[channel_index];
        const scale = scales.data[channel_index];
        const bias = B.data[channel_index];
        const x = input.data[flat_index];

        // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
        // Calculate inverse std dev for potentially better stability/performance
        const std_dev_inv = 1.0 / @sqrt(variance + epsilon);
        const normalized = (x - mean) * std_dev_inv;
        const result = normalized * scale + bias;

        // Assign final result, casting to output type T if needed.
        output.data[flat_index] = @as(T, result); // Keep cast for generality
        // --- End Calculation ---

        // Increment multi-dimensional index and update flat_index efficiently
        d = dims - 1;
        while (true) {
            // No need to subtract old contribution, just calculate new flat_index if needed,
            // but since we iterate linearly, we can just increment flat_index.
            // However, the loop structure increments the multi-dim index first.
            // Let's stick to incrementing multi-dim index and recalculating flat_index or just incrementing.
            // Simple increment is fastest if iteration order matches memory layout.
            // flat_index += 1; // This assumes standard row-major iteration matches memory

            current_index[d] += 1;
            if (current_index[d] < input.shape[d]) {
                // If we didn't reset this dimension, we are done incrementing for this step
                break;
            }
            // Reset current dim and carry over to the next higher dimension
            current_index[d] = 0;

            if (d == 0) {
                // Overflowed the highest dimension, means we are done iterating.
                break :outer_loop;
            }
            d -= 1; // Move to the next dimension to carry the increment
        }
        // After breaking the inner loop, recalculate flat_index based on current_index
        // This is safer than assuming flat_index++.
        flat_index = 0;
        var dim_idx: usize = 0;
        while (dim_idx < dims) {
            flat_index += current_index[dim_idx] * strides[dim_idx];
            dim_idx += 1;
        }

        // Check if we finished the whole tensor (alternative exit condition)
        // This check might be redundant if the carry logic is correct.
        // if (flat_index >= input.size) goto end_loop; // Use input.size or output.size
    }
}

pub inline fn get_batchNormalization_output_shape(input_shape: []usize) ![]usize {
    return input_shape;
}
