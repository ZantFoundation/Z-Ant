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
} //

pub inline fn readScalarZP(comptime T: type, zp_any: anytype) i32 {
    _ = T;
    return readScalarZPInternal(zp_any);
} //

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
} //

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
} //
