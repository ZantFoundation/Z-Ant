const zant = @import("../../../zant.zig");

const TensorModule = zant.core.tensor;

pub fn tryConvLean(
    comptime T: type,
    input: *const TensorModule.Tensor(T),
    weight: *const TensorModule.Tensor(T),
    output: *TensorModule.Tensor(T),
    bias: ?[]const T,
    params: anytype,
) !bool {
    _ = input;
    _ = weight;
    _ = output;
    _ = bias;
    _ = params;
    return false;
}

pub fn resetTestHooks() void {}

pub fn cmsisUsed() bool {
    return false;
}

pub fn ethosUsed() bool {
    return false;
}
