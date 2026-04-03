const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

pub const AutoPadType = enum {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

// Calculate output shape for GlobalAveragePool operation
/// For GlobalAveragePool, output shape is (N, C, 1, 1, ...) where all spatial dimensions become 1
fn get_global_average_pool_output_shape(input_shape: []const usize) ![]usize {
    if (input_shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);

    // First two dimensions (batch_size, channels) remain the same
    output_shape[0] = input_shape[0]; // N (batch size)
    output_shape[1] = input_shape[1]; // C (channels)

    // All spatial dimensions become 1
    for (2..input_shape.len) |i| {
        output_shape[i] = 1;
    }

    return output_shape;
}

/// GlobalAveragePool function following ONNX specification
/// Applies average pooling across the values in the same channel
/// This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor
/// The output tensor has the same rank as the input, with first two dimensions (N x C) unchanged
/// and all other dimensions set to 1
/// https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
pub fn globalAveragePool(
    comptime T: type,
    input: *Tensor(T),
) !Tensor(T) {
    // Validate input tensor has at least 2 dimensions (N, C)
    if (input.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output shape
    const output_shape = try get_global_average_pool_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform global average pooling
    try lean_globalAveragePool(T, input, &output);

    return output;
}

/// Lean version of GlobalAveragePool that operates on pre-allocated output tensor
/// This function performs the actual computation without allocating output tensor
/// https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
pub fn lean_globalAveragePool(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
) !void {
    // Validate input dimensions
    if (input.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    // Validate output shape
    if (output.shape.len != input.shape.len) {
        return TensorMathError.ShapeMismatch;
    }

    // Check that output has correct shape
    if (output.shape[0] != input.shape[0] or output.shape[1] != input.shape[1]) {
        return TensorMathError.ShapeMismatch;
    }

    // Check that all spatial dimensions in output are 1
    for (2..output.shape.len) |i| {
        if (output.shape[i] != 1) {
            return TensorMathError.ShapeMismatch;
        }
    }

    const batch_size = input.shape[0];
    const channels = input.shape[1];

    // Calculate spatial dimensions size
    var spatial_size: usize = 1;
    for (2..input.shape.len) |i| {
        spatial_size *= input.shape[i];
    }

    // If there are no spatial dimensions, just copy the input
    if (spatial_size == 0) {
        spatial_size = 1;
    }

    // Perform global average pooling for each batch and channel
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            var sum: T = 0;

            // Calculate the starting index for this batch and channel
            const batch_offset = n * channels * spatial_size;
            const channel_offset = c * spatial_size;
            const base_idx = batch_offset + channel_offset;

            // Sum all spatial elements for this batch and channel
            for (0..spatial_size) |spatial_idx| {
                const input_idx = base_idx + spatial_idx;
                sum += input.data[input_idx];
            }

            // Calculate average and store in output
            const average = sum / @as(T, @floatFromInt(spatial_size));

            // Calculate output index (all spatial dimensions are 1)
            const output_batch_offset = n * channels * 1; // spatial_size is 1 for output
            const output_channel_offset = c * 1;
            const output_idx = output_batch_offset + output_channel_offset;

            output.data[output_idx] = average;
        }
    }
}

// Alternative implementation with more explicit indexing for complex tensor layouts
pub fn lean_globalAveragePooling_explicit(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
) !void {

    // Validate input dimensions
    if (input.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    // Validate output shape matches expected global average pooling output
    if (output.shape.len != input.shape.len) {
        return TensorMathError.ShapeMismatch;
    }

    const batch_size = input.shape[0];
    const channels = input.shape[1];

    // Calculate spatial dimensions
    var spatial_dims: []usize = try pkg_allocator.alloc(usize, input.shape.len - 2);
    defer pkg_allocator.free(spatial_dims);

    var spatial_size: usize = 1;
    for (2..input.shape.len) |i| {
        spatial_dims[i - 2] = input.shape[i];
        spatial_size *= input.shape[i];
    }

    // Process each batch and channel combination
    for (0..batch_size) |batch| {
        for (0..channels) |channel| {
            var sum: T = 0;
            var count: usize = 0;

            // Iterate through all spatial positions
            var spatial_indices = try pkg_allocator.alloc(usize, spatial_dims.len);
            defer pkg_allocator.free(spatial_indices);

            // Initialize spatial indices to 0
            @memset(spatial_indices, 0);

            // Iterate through all spatial positions using nested loops approach
            var done = false;
            while (!done) {
                // Calculate linear index for current position
                var input_idx: usize = batch * channels;
                for (2..input.shape.len) |dim| {
                    input_idx *= input.shape[dim];
                }
                input_idx += channel;
                for (2..input.shape.len) |dim| {
                    input_idx *= input.shape[dim];
                }

                // Add contribution from this spatial position
                var spatial_offset: usize = 0;
                var stride: usize = 1;
                var dim_idx = spatial_dims.len;
                while (dim_idx > 0) {
                    dim_idx -= 1;
                    spatial_offset += spatial_indices[dim_idx] * stride;
                    stride *= spatial_dims[dim_idx];
                }

                // Calculate final input index
                const final_input_idx = batch * channels * spatial_size + channel * spatial_size + spatial_offset;
                sum += input.data[final_input_idx];
                count += 1;

                // Increment spatial indices (like odometer)
                var carry = true;
                var idx: usize = spatial_dims.len;
                while (idx > 0 and carry) {
                    idx -= 1;
                    spatial_indices[idx] += 1;
                    if (spatial_indices[idx] < spatial_dims[idx]) {
                        carry = false;
                    } else {
                        spatial_indices[idx] = 0;
                    }
                }
                done = carry;
            }

            // Calculate average and store in output
            const average = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;

            // Calculate output index (batch, channel, all spatial dims = 1)
            const output_idx: usize = batch * channels + channel;
            output.data[output_idx] = average;
        }
    }
}
