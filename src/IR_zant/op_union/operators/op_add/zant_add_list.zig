const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;

/// Returns a Tensor with the same shape as the input tensors, where each element is the sum of all tensors at that location
pub fn add_list(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType)) !Tensor(outputType) {
    if (tensors.len == 0) return error.EmptyTensorList;
    if (tensors.len == 1) {
        var out_tensor = try Tensor(outputType).fromShape(tensors[0].allocator, tensors[0].shape);
        for (0..tensors[0].data.len) |i| {
            out_tensor.data[i] = tensors[0].data[i];
        }
        return out_tensor;
    }

    // Use first tensor as reference for size and shape checks
    const ref_tensor = tensors[0];

    // Check all tensors have same size
    for (tensors[1..]) |t| {
        if (t.size != ref_tensor.size) return error.InputTensorDifferentSize;
    }

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return error.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return error.TooSmallOutputType;
    }

    var out_tensor = try Tensor(outputType).fromShape(ref_tensor.allocator, ref_tensor.shape);
    try add_list_lean(inputType, outputType, tensors, &out_tensor);

    return out_tensor;
}

pub inline fn add_list_lean(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    if (tensors.len == 0) return error.EmptyTensorList;

    // Initialize output with first tensor
    for (0..tensors[0].data.len) |i| {
        outputTensor.data[i] = tensors[0].data[i];
    }

    // Add remaining tensors
    for (tensors[1..]) |t| {
        for (0..t.data.len) |i| {
            outputTensor.data[i] += t.data[i];
        }
    }
}
