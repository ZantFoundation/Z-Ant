const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

// Followint the onnx standard
// https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
//Saturation is done according to:
// uint16: [0, 65535]
// int16: [-32768, 32767]
// uint8: [0, 255]
// int8: [-128, 127]
// uint4: [0, 15]
// int4: [-8, 7]

pub fn quantizeLinear(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    comptime ZeroPointType: anytype,
    x: *Tensor(InputType), //T1
    y_scale: *Tensor(InputType), //T2
    y_zero_point: ?*Tensor(ZeroPointType), //T3
    axis: i32,
    block_size: i32,
    // output_dtype: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // precision: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // saturate: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
) !Tensor(OutputType) {

    // Create output tensor shape copy
    var out = try Tensor(OutputType).fromShape(x.allocator, x.shape);

    // Dispatch to lean function
    try quantizeLinear_lean(
        InputType,
        OutputType,
        ZeroPointType,
        x,
        y_scale,
        y_zero_point,
        axis,
        block_size,
        // output_dtype,
        // precision,
        // saturate,
        &out,
    );
    return out;
}

pub inline fn quantizeLinear_lean(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    comptime _: anytype, // ZeroPointType unused due to anytype y_zero_point
    x: *Tensor(InputType), //T1
    y_scale: *Tensor(f32), //T2 - Scale is always f32
    y_zero_point: anytype, //T3 - Accept any tensor type for zero_point (can be null)
    axis: i32,
    block_size: i32,
    // output_dtype: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // precision: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // saturate: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    y: *Tensor(OutputType), //T4
) !void {
    // const is_perTensor = (axis == -1 or (axis == @as(i32, @intCast(x.shape.len - 1)) and block_size == 0));
    const N = x.size;

    if (N == 0) return;

    // quantization formula: y = saturate((x / y_scale) + y_zero_point)
    // three supported quantization granularities.
    //      Per-tensor (per-layer) quantization: y_scale is a scalar.
    //      Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape (D0, ..., Di, ..., Dn) and axis=i, y_scale is a 1-D tensor of length Di.
    //      Blocked quantization: The scale’s shape is identical to the input’s shape, except for one dimension, in which blocking is performed. Given x shape (D0, ..., Di, ..., Dn), axis=i, and block size B: y_scale shape is (D0, ..., ceil(Di/B), ..., Dn).

    const rank = @as(i32, @intCast(x.shape.len));

    // if (y_zero_point.shape.len != y_scale.shape.len) return error.Invalid_zeroPoint_or_scale;
    // for (0..y_zero_point.shape.len) |i| {
    //     if (y_zero_point.data[i] != y_scale.data[i]) return error.Invalid_zeroPoint_or_scale;
    // }

    const is_perTensor_lean: bool = y_scale.shape.len == 1 and y_scale.data.len == 1;
    const is_perAxis: bool = !is_perTensor_lean and y_scale.data.len == x.shape[@as(usize, @intCast(axis))] and y_scale.shape.len == 1;
    const is_perBlock: bool = false;

    // DEBUG: input stats
    var min_x: f32 = std.math.inf(f32);
    var max_x: f32 = -std.math.inf(f32);
    var sum_x: f64 = 0;
    for (0..N) |i| {
        const xv: f32 = if (@typeInfo(@TypeOf(x.data[0])) == .int) @as(f32, @floatFromInt(x.data[i])) else @as(f32, x.data[i]);
        if (xv < min_x) min_x = xv;
        if (xv > max_x) max_x = xv;
        sum_x += xv;
    }
    _ = @as(f32, @floatCast(sum_x / @as(f64, @floatFromInt(N))));
    _ = if (@TypeOf(y_zero_point) == @TypeOf(null)) 0 else blk: {
        if (@typeInfo(@TypeOf(y_zero_point)) == .optional) {
            break :blk if (y_zero_point == null) 0 else y_zero_point.?.data[0];
        } else {
            break :blk y_zero_point.data[0];
        }
    };

    //check if is_perBlock
    if (!is_perTensor_lean and !is_perAxis) {
        if (y_scale.shape.len != x.shape.len) return error.Invalid_perBlock_scale; // The scale’s shape is identical to the input’s shape -> same lenght

        for (0..y_scale.shape.len) |i| { //The scale’s shape is identical to the input’s shape
            if (i == axis) { //dimension where bloking is performed
                if (block_size <= 0) return error.Invalid_block_size;
                if (y_scale.shape[i] != @divFloor(@as(i32, @intCast(x.shape[i])), block_size)) return error.Invalid_scale_shape;
            } else {
                if (y_scale.shape[i] != x.shape[i]) return error.Invalid_scale_shape;
            }
        }
    }

    if (is_perBlock) {
        // Blocked quantization: different scale/zero_point for each block
        const axis_u = @as(usize, @intCast(axis));
        const block_size_u = @as(usize, @intCast(block_size));

        // Calculate strides for multi-dimensional indexing
        var x_strides: [4]usize = undefined; // assuming max 4 dimensions
        var scale_strides: [4]usize = undefined;

        // Calculate strides for x tensor
        x_strides[rank - 1] = 1;
        for (0..rank - 1) |i| {
            const idx = rank - 2 - i;
            x_strides[idx] = x_strides[idx + 1] * x.shape[idx + 1];
        }

        // Calculate strides for scale tensor
        scale_strides[rank - 1] = 1;
        for (0..rank - 1) |i| {
            const idx = rank - 2 - i;
            scale_strides[idx] = scale_strides[idx + 1] * y_scale.shape[idx + 1];
        }

        // Iterate through all elements
        for (0..N) |i| {

            // Calculate multi-dimensional indices for x
            var x_indices: [8]usize = undefined;
            var remaining = i;
            for (0..rank) |dim| {
                x_indices[dim] = remaining / x_strides[dim];
                remaining %= x_strides[dim];
            }

            // Calculate corresponding scale indices (block index for axis dimension)
            var scale_indices: [8]usize = undefined;
            for (0..rank) |dim| {
                if (dim == axis_u) {
                    scale_indices[dim] = x_indices[dim] / block_size_u;
                } else {
                    scale_indices[dim] = x_indices[dim];
                }
            }

            // Calculate flat index for scale
            var scale_idx: usize = 0;
            for (0..rank) |dim| {
                scale_idx += scale_indices[dim] * scale_strides[dim];
            }

            // Apply quantization formula
            const scale_val = y_scale.data[scale_idx];
            const zero_point_val = if (@TypeOf(y_zero_point) == @TypeOf(null)) 0 else blk: {
                if (@typeInfo(@TypeOf(y_zero_point)) == .optional) {
                    break :blk if (y_zero_point == null) 0 else y_zero_point.?.data[scale_idx];
                } else {
                    break :blk y_zero_point.data[scale_idx];
                }
            };

            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    } else if (is_perAxis) {
        // Per-axis quantization: different scale/zero_point for each slice along the axis
        const axis_u = @as(usize, @intCast(axis));
        const axis_size = x.shape[axis_u];

        // Calculate stride for the axis dimension
        var axis_stride: usize = 1;
        for (axis_u + 1..x.shape.len) |i| {
            axis_stride *= x.shape[i];
        }

        // Calculate outer stride (elements between different axis slices)
        // var outer_stride: usize = axis_stride * axis_size;

        for (0..N) |i| {
            // Calculate which axis index this element belongs to
            const axis_idx = (i / axis_stride) % axis_size;

            // Apply quantization formula
            const scale_val = y_scale.data[axis_idx];
            const zero_point_val = if (@TypeOf(y_zero_point) == @TypeOf(null)) 0 else blk: {
                if (@typeInfo(@TypeOf(y_zero_point)) == .optional) {
                    break :blk if (y_zero_point == null) 0 else y_zero_point.?.data[axis_idx];
                } else {
                    break :blk y_zero_point.data[axis_idx];
                }
            };

            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    } else if (is_perTensor_lean) {
        // Per-tensor quantization: same scale/zero_point for all elements
        const scale_val = y_scale.data[0];
        const zero_point_val = if (@TypeOf(y_zero_point) == @TypeOf(null)) 0 else blk: {
            if (@typeInfo(@TypeOf(y_zero_point)) == .optional) {
                break :blk if (y_zero_point == null) 0 else y_zero_point.?.data[0];
            } else {
                break :blk y_zero_point.data[0];
            }
        };

        for (0..N) |i| {
            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    }

    // DEBUG: output stats and samples
    var min_y: f32 = std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);
    var sum_y: f64 = 0;
    for (0..N) |i| {
        const yv: f32 = if (@typeInfo(@TypeOf(y.data[0])) == .int) @as(f32, @floatFromInt(y.data[i])) else @as(f32, y.data[i]);
        if (yv < min_y) min_y = yv;
        if (yv > max_y) max_y = yv;
        sum_y += yv;
    }
    _ = @as(f32, @floatCast(sum_y / @as(f64, @floatFromInt(N))));
    const sample = @min(N, 6);
    for (0..sample) |i| {
        _ = if (@typeInfo(@TypeOf(x.data[0])) == .int) @as(f32, @floatFromInt(x.data[i])) else @as(f32, x.data[i]);
        _ = if (@typeInfo(@TypeOf(y.data[0])) == .int) @as(f32, @floatFromInt(y.data[i])) else @as(f32, y.data[i]);
    }
}

inline fn quantize(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    inputData: InputType,
    scale: f32, // Scale is always f32
    zp: OutputType,
) !OutputType {
    const input_f32 = if (InputType == f32) inputData else @as(f32, @floatFromInt(inputData));
    const scaled: f32 = input_f32 / scale;
    // Use standard rounding (round half to even) - Zig's @round is ONNX-compliant
    const rounded = @round(scaled);
    const quantized = rounded + @as(f32, @floatFromInt(zp));

    return saturate(f32, OutputType, quantized);
}

// Helper function to saturate values according to output type
inline fn saturate(comptime InputType: type, comptime OutputType: type, value: InputType) OutputType {
    const info = @typeInfo(OutputType);
    if (info == .int) {
        @setEvalBranchQuota(10000);
        const min_val: InputType = std.math.minInt(OutputType);
        const max_val: InputType = std.math.maxInt(OutputType);
        return @as(OutputType, @intFromFloat(std.math.clamp(value, min_val, max_val)));
    } else {
        // For float types, just convert directly
        return @as(OutputType, value);
    }
}
