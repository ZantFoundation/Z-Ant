const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

//TODO aggiungi tests

//TODO scalar optimization
pub fn get_pow_output_shape(comptime T: type, base: *const Tensor(T), exp: *const Tensor(T)) ![]usize {

    //broadcast
    const len1 = base.shape.len;
    const len2 = exp.shape.len;
    const maxLen = @max(len1, len2);

    //creating the output
    const output = try pkg_allocator.alloc(usize, maxLen);
    errdefer pkg_allocator.free(output);

    //check if one input is scalar
    //if (len1 == 0 or len2 == 0) {}

    //setting offsets
    const offset1: usize = maxLen - len1;
    const offset2: usize = maxLen - len2;

    //filling output shape
    var pos: usize = 0;
    while (pos < maxLen) : (pos += 1) {
        const dim1: usize = if (pos < offset1) 1 else base.shape[pos - offset1];
        const dim2: usize = if (pos < offset2) 1 else exp.shape[pos - offset2];

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }

        output[pos] = if (dim1 >= dim2) dim1 else dim2;
    }

    return output;
}

pub fn pow(comptime T: type, base: *const Tensor(T), exp: *const Tensor(T)) !Tensor(T) {
    const outputShape = try get_pow_output_shape(T, base, exp);
    defer pkg_allocator.free(outputShape);

    var output = try Tensor(T).fromShape(&pkg_allocator, outputShape);
    errdefer output.deinit();

    try pow_lean(T, base, exp, &output);

    return output;
}

//TODO scalar optimization
pub fn pow_lean(comptime T: type, baseTensor: *const Tensor(T), expTensor: *const Tensor(T), output: *Tensor(T)) !void {
    const len1 = baseTensor.data.len;
    const len2 = expTensor.data.len;

    var ia: usize = 0;
    var ib: usize = 0;
    var i: usize = 0;

    while (i < output.data.len) : (i += 1) {
        if (ia >= len1) ia = 0;
        if (ib >= len2) ib = 0;

        output.data[i] = std.math.pow(T, baseTensor.data[ia], expTensor.data[ib]);

        //incremenet
        ia += 1;
        ib += 1;
    }
}
