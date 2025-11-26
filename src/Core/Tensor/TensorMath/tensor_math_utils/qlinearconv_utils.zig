const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Contains a series of utility functions for QLinearConv operations.
// These functions handle reading zero-points and scales from various
// representations, as well as quantizing multipliers for CMSIS-NN.
// They are designed to be flexible and work with different tensor shapes
// and data types.
// These functions are shared and used by both Zant's native QLinearConv
// implementation and the CMSIS-NN backend.

// Helper function to quantize multiplier for CMSIS-NN
pub fn quantizeMultiplier(scale: f32, multiplier: *i32, shift: *i32) void {
    if (scale == 0.0) {
        multiplier.* = 0;
        shift.* = 0;
        return;
    }

    var sig = scale;
    var exp: i32 = 0;

    // Normalize to [0.5, 1.0) range
    while (sig >= 1.0) {
        sig /= 2.0;
        exp += 1;
    }
    while (sig < 0.5) {
        sig *= 2.0;
        exp -= 1;
    }

    // Convert to fixed point representation
    const fixed_point_multiplier = @as(i32, @intFromFloat(@round(sig * (1 << 31))));

    multiplier.* = fixed_point_multiplier;
    shift.* = exp;
}

pub inline fn readScalarZP(comptime T: type, zp_any: anytype) i32 {
    _ = T;
    return readScalarZPInternal(zp_any);
}

fn readScalarZPInternal(zp_any: anytype) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readScalarZPInternal(zp_any.*),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                break :blk @as(i32, @intCast(zp_any[0]));
            },
            .many, .c => blk: {
                // Treat bare pointers as single-value buffers and read the first element.
                break :blk @as(i32, @intCast(zp_any[0]));
            },
        },
        .optional => if (zp_any) |payload| readScalarZPInternal(payload) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(data[0]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

inline fn selectChannelIndex(len: usize, channel: usize) usize {
    if (len <= 1) return 0;
    return if (channel < len) channel else len - 1;
}

pub inline fn readPerChannelZP(zp_any: anytype, m: usize, M: usize) i32 {
    _ = M;
    return readPerChannelZPInternal(zp_any, m);
}

fn readPerChannelZPInternal(zp_any: anytype, m: usize) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readPerChannelZPInternal(zp_any.*, m),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                const idx = selectChannelIndex(zp_any.len, m);
                break :blk @as(i32, @intCast(zp_any[idx]));
            },
            .many, .c => @compileError("unsupported zero-point pointer representation"),
        },
        .optional => if (zp_any) |payload| readPerChannelZPInternal(payload, m) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.array.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.vector.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            const idx = selectChannelIndex(data.len, m);
            break :blk @as(i32, @intCast(data[idx]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

pub inline fn clampToI8(v: anytype) i8 {
    const val_i32: i32 = @as(i32, @intCast(v));
    const clamped = std.math.clamp(val_i32, std.math.minInt(i8), std.math.maxInt(i8));
    return @as(i8, @intCast(clamped));
}

// IT USES THE NHWC CONVENTION TO COMPUTE OUTPUT SHAPE
pub fn nhwc_calculateOutputShape(
    comptime T: type,
    allocator: std.mem.Allocator,
    input_shape: []const usize, // NHWC: [N, H, W, C]
    weight_shape: []const usize, // OHWI: [O, kH, kW, C]
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    auto_pad: ?[]const u8,
) ![]usize {
    _ = T; // unused

    if (input_shape.len != 4 or weight_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const batch_size = input_shape[0];
    const in_height = input_shape[1];
    const in_width = input_shape[2];
    // const in_channels = input_shape[3];  // not needed here

    // OHWI extraction
    const out_channels = weight_shape[0];
    const kernel_height = weight_shape[1];
    const kernel_width = weight_shape[2];
    // const weight_in_channels = weight_shape[3];

    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;

    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Effective kernel size
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            // No padding
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or
            std.mem.eql(u8, pad_mode, "SAME_LOWER"))
        {
            // SAME padding, output = ceil(input/stride)
            const out_height = (in_height + stride_h - 1) / stride_h;
            const out_width = (in_width + stride_w - 1) / stride_w;

            const total_pad_h = @max(0, (out_height - 1) * stride_h + dilated_kernel_h - in_height);
            const total_pad_w = @max(0, (out_width - 1) * stride_w + dilated_kernel_w - in_width);

            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = total_pad_h / 2;
                pad_h_end = total_pad_h - pad_h_begin;

                pad_w_begin = total_pad_w / 2;
                pad_w_end = total_pad_w - pad_w_begin;
            } else {
                // SAME_LOWER
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;

                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        } else if (!std.mem.eql(u8, pad_mode, "NOTSET")) {
            return TensorMathError.InvalidPadding;
        }
    }

    // Explicit pads override auto_pad
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

    const padded_height = in_height + pad_h_begin + pad_h_end;
    const padded_width = in_width + pad_w_begin + pad_w_end;

    if (padded_height < dilated_kernel_h or padded_width < dilated_kernel_w) {
        return TensorMathError.InvalidDimensions;
    }

    const out_height =
        (padded_height - dilated_kernel_h) / stride_h + 1;
    const out_width =
        (padded_width - dilated_kernel_w) / stride_w + 1;

    // ================================
    // Return NHWC shape: [N, H_out, W_out, C_out]
    // ================================
    const output_shape = try allocator.alloc(usize, 4);
    output_shape[0] = batch_size;
    output_shape[1] = out_height;
    output_shape[2] = out_width;
    output_shape[3] = out_channels;

    return output_shape;
}
