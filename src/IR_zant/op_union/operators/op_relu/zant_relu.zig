const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

/// ReLU (Rectified Linear Unit).
/// It outputs the input directly if it's positive, but returns zero for any negative input.
pub inline fn ReLU_standard(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    // Allocate output with same shape
    var output = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);

    try lean_ReLU(T, tensor, &output);

    return output;
}

/// In-place lean ReLU supporting quantized and standard tensors
pub inline fn lean_ReLU(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    //apply ReLU
    //OSS: can be improved, see how did I parallelized CPU Tensor Sum
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] > 0) output_tensor.data[i] = input_tensor.data[i];
    }
}
