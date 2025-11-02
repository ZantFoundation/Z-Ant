const zant = @import("../../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;

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
    _ = input;
    _ = weight;
    _ = output;
    _ = bias;
    _ = stride;
    _ = pads;
    _ = dilations;
    _ = group;
    _ = auto_pad;

    // Quantization

    // Call CMSIS

    // Dequantization

    return false; // empty stub (to be implemented)
}
