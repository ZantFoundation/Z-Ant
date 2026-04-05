const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub const utils = @import("utils_pow.zig");

pub fn pow(comptime T: type, comptime T1: type, base: *Tensor(T), exp: *Tensor(T1)) !Tensor(T) {

    //check for unsupported types at compile time
    comptime {
        const isSupported = switch (T) {
            f16, f32, f64, i32, i64 => true,
            else => false,
        };
        const isSupported2 = switch (T1) {
            f16, f32, f64, i32, i64 => true,
            else => false,
        };
        if (!isSupported or !isSupported2) return error.InvalidDataType;
    }

    const outputShape = try utils.get_pow_output_shape(T, T1, base, exp);
    defer pkg_allocator.free(outputShape);

    var output = try Tensor(T).fromShape(&pkg_allocator, outputShape);
    errdefer output.deinit();

    try pow_lean(T, T1, base, exp, &output);

    return output;
}

pub fn pow_lean(comptime T: type, comptime T1: type, baseTensor: *Tensor(T), expTensor: *Tensor(T1), output: *Tensor(T)) !void {
    for (0..output.size) |idx| {
        const coords = try utils.indexToCoords(idx, output.shape);
        defer pkg_allocator.free(coords);

        const base_idx = utils.getBroadcastIndex(coords, baseTensor.shape, output.shape);
        const exp_idx = utils.getBroadcastIndex(coords, expTensor.shape, output.shape);

        const baseValue = baseTensor.data[base_idx];
        const expValueCasted = utils.castToType(T, T1, expTensor.data[exp_idx]);

        if (baseValue == 0 and expValueCasted < 0) return error.DivisionError;

        const result = std.math.pow(T, baseValue, expValueCasted);

        output.data[idx] = result;
    }
}
