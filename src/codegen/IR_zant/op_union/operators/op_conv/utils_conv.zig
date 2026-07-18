const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub fn get_conv_output_shape(
    comptime T: type,
    allocator: std.mem.Allocator,
    input_shape: []const usize, // [N, C, H, W]
    weight_shape: []const usize, // [M, C/group, kH, kW]
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    // group: ?usize,
    auto_pad: ?[]const u8,
) ![]usize {
    _ = T; // Suppress unused parameter warning
    if (input_shape.len != 4 or weight_shape.len != 4) {
        return error.InvalidDimensions;
    }
    const batch_size = input_shape[0];
    const in_height = input_shape[2];
    const in_width = input_shape[3];
    const out_channels = weight_shape[0];
    const kernel_height = weight_shape[2];
    const kernel_width = weight_shape[3];

    // Set defaults
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Calculate effective kernel size with dilation
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            // No padding - already initialized to 0
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            // For SAME padding, output size should be ceil(input_size / stride)
            const out_height = (in_height + stride_h - 1) / stride_h;
            const out_width = (in_width + stride_w - 1) / stride_w;
            const total_pad_h = if (out_height * stride_h + dilated_kernel_h > in_height)
                (out_height - 1) * stride_h + dilated_kernel_h - in_height
            else
                0;
            const total_pad_w = if (out_width * stride_w + dilated_kernel_w > in_width)
                (out_width - 1) * stride_w + dilated_kernel_w - in_width
            else
                0;
            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = total_pad_h / 2;
                pad_h_end = total_pad_h - pad_h_begin;
                pad_w_begin = total_pad_w / 2;
                pad_w_end = total_pad_w - pad_w_begin;
            } else { // SAME_LOWER
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;
                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        } else if (!std.mem.eql(u8, pad_mode, "NOTSET")) {
            return error.InvalidPadding;
        }
    }

    // Use explicit padding if provided
    if (pads) |p| {
        if (p.len >= 4) {
            pad_h_begin = p[0];
            pad_w_begin = p[1];
            pad_h_end = p[2];
            pad_w_end = p[3];
        } else if (p.len == 2) {
            pad_h_begin = p[0];
            pad_h_end = p[0];
            pad_w_begin = p[1];
            pad_w_end = p[1];
        }
    }

    // Calculate output dimensions
    const padded_height = in_height + pad_h_begin + pad_h_end;
    const padded_width = in_width + pad_w_begin + pad_w_end;

    if (padded_height < dilated_kernel_h or padded_width < dilated_kernel_w) {
        return error.InvalidDimensions;
    }

    const out_height = (padded_height - dilated_kernel_h) / stride_h + 1;
    const out_width = (padded_width - dilated_kernel_w) / stride_w + 1;

    // Allocate and return slice
    const output_shape = try allocator.alloc(usize, 4);
    output_shape[0] = batch_size;
    output_shape[1] = out_channels;
    output_shape[2] = out_height;
    output_shape[3] = out_width;

    return output_shape;
}

pub const conv_clip_lean = @import("../../fused/fused_conv_clip/zant_conv_clip.zig").conv_clip_lean;
