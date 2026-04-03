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

pub fn dequantizeLinear(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType), //T1
    x_scale: *const Tensor(OutputType), //T2
    x_zero_point: ?*const Tensor(ZeroPointType), //T1
    axis: i32,
    block_size: i32,
) !Tensor(OutputType) {
    var out = try Tensor(OutputType).fromShape(x.allocator, x.shape);

    try dequantizeLinear_lean(
        InputType,
        OutputType,
        ZeroPointType,
        x,
        x_scale,
        x_zero_point,
        axis,
        block_size,
        &out,
    );
    return out;
}

pub inline fn dequantizeLinear_lean(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    comptime _: anytype, // ZeroPointType unused due to anytype x_zero_point
    x: *Tensor(InputType), //T1
    x_scale: *const Tensor(OutputType), //T2
    x_zero_point: anytype, //T1 - Accept any tensor type for zero_point (can be null)
    axis: i32,
    block_size: i32,
    // output_dtype: i32, only used when parsing, see IR_zant/op_union_operators/op_dequantizeLinear
    // precision: i32, only used when parsing, see IR_zant/op_union_operators/op_dequantizeLinear
    // saturate: i32, only used when parsing, see IR_zant/op_union_operators/op_dequantizeLinear
    y: *Tensor(OutputType), //T2
) !void {
    const N: usize = @min(x.data.len, y.data.len);
    const rank = x.shape.len;
    const is_axis = x_scale.data.len > 1;
    const is_block = block_size > 0;
    y.size = x.data.len;

    // DEBUG: print config
    _ = if (@TypeOf(x_zero_point) == @TypeOf(null)) 0 else x_zero_point.data[0];

    // Normalize axis
    const normalized_axis: usize = if (axis < 0)
        @intCast(@as(i32, @intCast(rank)) + axis)
    else
        @intCast(axis);

    // Pre-compute strides for the target axis if needed
    var axis_stride: usize = 1;
    if (is_axis or is_block) {
        for (normalized_axis + 1..rank) |i| {
            axis_stride *= x.shape[i];
        }
    }

    for (0..N) |i| {
        var idx: usize = 0;
        var zp_val: i32 = 0;

        if (@TypeOf(x_zero_point) != @TypeOf(null)) {
            if (is_block) {
                // For blocked quantization
                const dim_size = x.shape[normalized_axis];
                const axis_coord = (i / axis_stride) % dim_size;
                const block_idx = axis_coord / @as(usize, @intCast(block_size));

                // Calculate the linear index in the scale/zero_point tensor
                // by mapping the multi-dimensional coordinate
                var linear_idx: usize = 0;
                var temp_i = i;
                var stride_scale: usize = 1;

                for (0..rank) |dim| {
                    const dim_idx = rank - 1 - dim;
                    const coord = temp_i % x.shape[dim_idx];
                    temp_i /= x.shape[dim_idx];

                    if (dim_idx == normalized_axis) {
                        // Use block index for the axis dimension
                        linear_idx += block_idx * stride_scale;
                    } else {
                        linear_idx += coord * stride_scale;
                    }

                    if (dim_idx == normalized_axis) {
                        stride_scale *= (dim_size + @as(usize, @intCast(block_size)) - 1) / @as(usize, @intCast(block_size));
                    } else {
                        stride_scale *= x.shape[dim_idx];
                    }
                }
                idx = linear_idx;
                zp_val = @as(i32, x_zero_point.data[idx]);
            } else if (is_axis) {
                // For per-axis quantization
                const axis_coord = (i / axis_stride) % x.shape[normalized_axis];
                idx = axis_coord;
                zp_val = @as(i32, x_zero_point.data[idx]);
            } else {
                // Per-tensor quantization
                idx = 0;
                zp_val = @as(i32, x_zero_point.data[0]);
            }
        } else {
            // No zero_point specified → default 0
            zp_val = 0;
            idx = 0;
        }

        const scale: OutputType = x_scale.data[idx];
        const xval: InputType = x.data[i];
        const deq = if (@typeInfo(InputType) == .int)
            @as(OutputType, @floatFromInt(@as(i32, xval) - zp_val)) * scale
        else
            @as(OutputType, xval - @as(InputType, @floatFromInt(zp_val))) * scale;

        y.data[i] = deq;

        // Debug first 10 assignments
        if (i < 10) {}
    }

    // DEBUG: output stats
    var min_y: f32 = std.math.inf(f32);
    var max_y: f32 = -std.math.inf(f32);
    var sum_y: f64 = 0;
    for (0..N) |i| {
        const v = @as(f32, y.data[i]);
        if (v < min_y) min_y = v;
        if (v > max_y) max_y = v;
        sum_y += v;
    }
    _ = @as(f32, @floatCast(sum_y / @as(f64, @floatFromInt(N))));
}
