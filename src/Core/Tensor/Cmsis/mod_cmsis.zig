const zant = @import("../../../zant.zig");
const build_options = @import("build_options");

const Tensor = zant.core.tensor.Tensor;
const TensorModule = zant.core.tensor;

const convolution_wrapper = @import("wrappers/convolution.zig");

pub fn tryConv(
    comptime T: type,
    input: *const Tensor(T), // X: Input tensor [N, C, H, W]
    weight: *const Tensor(T), // W: Weight tensor [M, C/group, kH, kW]
    output: *Tensor(T), // Y: Output tensor [N, M, oH, oW]
    bias: ?*const Tensor(T), // B: Optional bias tensor [M]
    stride: ?[]const usize, // Stride along each spatial axis
    pads: ?[]const usize, // Padding [h_begin, w_begin, h_end, w_end]
    dilations: ?[]const usize, // Dilation along each spatial axis
    group: ?usize, // Number of groups (default 1)
    auto_pad: ?[]const u8, // NOTSET, VALID, SAME_UPPER, SAME_LOWER
) !bool {
    if (@hasDecl(build_options, "enabl_cmsis") and build_options.enable_cmsis) {
        return try convolution_wrapper.tryConvLean(T, input, weight, output, bias, stride, pads, dilations, group, auto_pad);
    }
    return false;
}

pub fn cmsisUsed() bool {
    return (@hasDecl(build_options, "enabl_cmsis") and build_options.enable_cmsis);
}
