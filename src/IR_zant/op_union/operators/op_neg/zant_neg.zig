const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;

const pkg_allocator = zant.utils.allocator.allocator;

const get_neg_output_shape = @import("utils_neg.zig").get_neg_output_shape;

/// Computes element-wise negation, multiplying each element by -1
/// This is the ONNX Neg operation: Y = -X
pub fn neg(comptime T: type, tensor: *Tensor(T)) !Tensor(T) {
    const neg_shape = get_neg_output_shape(tensor.shape);
    defer tensor.allocator.free(neg_shape);

    var neg_tensor = try Tensor(T).fromShape(tensor.allocator, neg_shape);

    try neg_lean(T, tensor, &neg_tensor);
    return neg_tensor;
}

/// Element-wise negation implementation (multiplies each element by -1)
pub fn neg_lean(comptime T: type, input: *Tensor(T), output: *Tensor(T)) !void {
    if (output.size != input.size) {
        return error.MismatchedShape;
    }

    for (0..input.size) |i| {
        output.data[i] = -input.data[i];
    }
}
