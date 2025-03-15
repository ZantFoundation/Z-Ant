const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

/// The Softmax activation function is used in multi-class classification tasks to convert
/// logits (raw output values) into probabilities that sum to 1.
/// Ideal for output layers in multi-class neural networks.
pub fn softmax(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {

    // TODO: add more robust checks
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    //compute
    try lean_softmax(T, tensor, &output_tensor);

    return output_tensor;
}

pub inline fn lean_softmax(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    const rows = input.shape[0];
    const cols = input.shape[1];

    var max_val: T = undefined;
    var sum_of_exp: T = 0.0;
    var val: T = undefined;

    // For each row
    for (0..rows) |i| {
        // Find the maximum value in the row to stabilize the computation
        max_val = input.data[i * cols];
        for (0..cols) |j| {
            val = input.data[i * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute stabilized exponentials and their sum
        sum_of_exp = 0.0;
        for (0..cols) |j| {
            val = input.data[i * cols + j] - max_val; // Stabilization
            val = @exp(val);
            output.data[i * cols + j] = val;
            sum_of_exp += val;
        }

        // Normalize to calculate the softmax
        for (0..cols) |j| {
            output.data[i * cols + j] /= sum_of_exp;
        }
    }
}

pub fn softmax_backward(comptime T: anytype, dL_dX: *Tensor(T), softmax_output: *Tensor(T)) !void {
    //checks
    if (dL_dX.size <= 0) return TensorError.ZeroSizeTensor;
    if (dL_dX.size != softmax_output.size) return TensorMathError.InputTensorDifferentSize;

    const dim = dL_dX.shape.len;
    const rows = dL_dX.shape[dim - 2];
    const cols = dL_dX.shape[dim - 1];

    // Make a copy of incoming gradients since we'll modify dL_dX
    var dL_dS = try dL_dX.copy();
    defer dL_dS.deinit();

    // For each sample in the batch
    for (0..rows) |i| {
        const row_offset = i * cols;
        // For each output neuron j
        for (0..cols) |j| {
            var sum: T = 0;
            const sj = softmax_output.data[row_offset + j];

            // Compute sum of dL/dS_k * S_k for all k
            for (0..cols) |k| {
                const sk = softmax_output.data[row_offset + k];
                const dL_dS_k = dL_dS.data[row_offset + k];
                if (j == k) {
                    sum += dL_dS_k * sk * (1 - sj); // Diagonal term
                } else {
                    sum += dL_dS_k * (-sk * sj); // Off-diagonal term
                }
            }
            dL_dX.data[row_offset + j] = sum;
        }
    }
}
