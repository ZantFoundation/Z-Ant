const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing concatenate operation for shape calculation and core logic
const concatenate = @import("lib_shape_math/op_concatenate.zig");

/// QLinearConcat operation following ONNX specification
/// Performs quantized tensor concatenation using linear quantization scheme
///
/// INPUTS (variable number):
/// - input_tensors: array of quantized input tensors (typically int8/uint8)
/// - input_scales: array of scale factors for each input quantization
/// - input_zero_points: array of zero points for each input quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
/// - axis: axis along which to concatenate
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(concat(dequantize(input1), dequantize(input2), ...), y_scale, y_zero_point)
pub fn qlinearconcat(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    input_tensors: []const *const Tensor(InputType),
    input_scales: []const *const Tensor(ScaleType),
    input_zero_points: []const *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    axis: isize,
) !Tensor(InputType) {
    // Input validation
    if (input_tensors.len == 0) return TensorMathError.EmptyTensorList;
    if (input_tensors.len != input_scales.len or input_tensors.len != input_zero_points.len) {
        return TensorMathError.InvalidDimensions;
    }
    if (y_scale.size != 1 or y_zero_point.size != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Validate all scale and zero point tensors are scalars
    for (input_scales) |scale| {
        if (scale.size != 1) return TensorMathError.InvalidDimensions;
    }
    for (input_zero_points) |zp| {
        if (zp.size != 1) return TensorMathError.InvalidDimensions;
    }

    // Calculate output shape using existing concatenate logic
    var input_shapes = try pkg_allocator.alloc([]const usize, input_tensors.len);
    defer {
        for (input_shapes) |shape| pkg_allocator.free(shape);
        pkg_allocator.free(input_shapes);
    }

    for (input_tensors, 0..) |input, i| {
        var shape = try pkg_allocator.alloc(usize, input.shape.len);
        for (input.shape, 0..) |dim, j| {
            shape[j] = if (dim < 0) 1 else @intCast(dim);
        }
        input_shapes[i] = shape;
    }

    const output_shape = try concatenate.get_concatenate_output_shape(input_shapes, axis);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized concatenation
    try lean_qlinearconcat(
        InputType,
        ScaleType,
        ZeroPointType,
        input_tensors,
        input_scales,
        input_zero_points,
        y_scale,
        y_zero_point,
        axis,
        &output,
    );

    return output;
}

/// Lean version of QLinearConcat that operates on pre-allocated output tensor
pub fn lean_qlinearconcat(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    input_tensors: []const *const Tensor(InputType),
    input_scales: []const *const Tensor(ScaleType),
    input_zero_points: []const *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    axis: isize,
    output: *Tensor(InputType),
) !void {
    // Skip computation for placeholder tensors
    if (input_tensors.len == 1 and input_tensors[0].size == 1 and output.size == 1 and
        input_tensors[0].shape.len == 1 and output.shape.len == 1)
    {
        const OutputType = @TypeOf(output.data[0]);
        output.data[0] = if (@typeInfo(OutputType) == .int) 0 else 0.0;
        return;
    }

    if (input_tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Normalize axis
    const rank = input_tensors[0].shape.len;
    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }
    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }
    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Get quantization parameters
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = y_zero_point.data[0];

    // DEBUG: QLinearConcat parameters
    if (false) { // DISABLED
        var debug_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(debug_buf[0..], "[QCONCAT_DEBUG] QLinearConcat axis={d}, num_inputs={d}\n  y_scale={}, y_zp={}\n", .{ concat_axis, input_tensors.len, y_scale_val, y_zero_point_val }) catch "Debug format error\n";
        debug_buf[msg.len] = 0;

        // Log input shapes and scales
        for (input_tensors, 0..) |input, i| {
            if (i < 3) { // Only first 3 inputs
                const input_scale_val = input_scales[i].data[0];
                const input_zp_val = input_zero_points[i].data[0];
                const msg2 = std.fmt.bufPrint(debug_buf[0..], "  Input[{d}]: shape=[{d},{d},{d},{d}], scale={}, zp={}\n", .{ i, input.shape[0], input.shape[1], input.shape[2], input.shape[3], input_scale_val, input_zp_val }) catch "Debug format error\n";
                debug_buf[msg2.len] = 0;
            }
        }
    }

    // Helper functions for type handling
    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    // Calculate slice sizes for efficient copying
    var num_slices: usize = 1;
    for (0..concat_axis_usize) |d| {
        num_slices *= output.shape[d];
    }

    var slice_size: usize = 1;
    if (concat_axis_usize + 1 < rank) {
        for ((concat_axis_usize + 1)..rank) |d| {
            slice_size *= output.shape[d];
        }
    } else {
        slice_size = 1;
    }

    // Initialize the offset for copying data into output
    var offset: usize = 0;

    // Iterate over each slice
    for (0..num_slices) |slice_idx| {
        for (input_tensors, 0..) |input, tensor_idx| {
            const input_scale_val = input_scales[tensor_idx].data[0];
            const input_zero_point_val = input_zero_points[tensor_idx].data[0];

            const concat_dim = input.shape[concat_axis_usize];
            const copy_size = concat_dim * slice_size;

            // Calculate the start indices in the source tensor
            const src_start = slice_idx * concat_dim * slice_size;
            const src_end = src_start + copy_size;

            // Check bounds for the source tensor's data
            if (src_end > input.size) {
                return TensorError.IndexOutOfBounds;
            }

            // Calculate the destination indices in output data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the output buffer
            if (dest_end > output.data.len) {
                return TensorError.IndexOutOfBounds;
            }

            // Dequantize and re-quantize each element
            for (src_start..src_end) |src_idx| {
                const dest_idx = dest_start + (src_idx - src_start);

                if (src_idx >= input.size) {
                    return TensorError.IndexOutOfBounds;
                }
                if (dest_idx >= output.data.len) {
                    return TensorError.IndexOutOfBounds;
                }

                // Dequantize input (always via f32)
                const input_dequant: f32 =
                    (asF32(InputType, input.data[src_idx]) - asF32(ZeroPointType, input_zero_point_val)) *
                    asF32(ScaleType, input_scale_val);

                // Re-quantize to output scale/zero-point
                const scaled_result: f32 =
                    (input_dequant / asF32(ScaleType, y_scale_val)) +
                    asF32(ZeroPointType, y_zero_point_val);

                // Clamp and store (QLinear outputs are integer-quantized)
                const OutputType = @TypeOf(output.data[0]);
                const min_val: f32 = @as(f32, @floatFromInt(std.math.minInt(OutputType)));
                const max_val: f32 = @as(f32, @floatFromInt(std.math.maxInt(OutputType)));
                const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

                const quantized_output = @as(OutputType, @intFromFloat(@round(clamped_result)));
                output.data[dest_idx] = quantized_output;

                // DEBUG: First few requantization operations
                if (false) { // DISABLED
                    if (slice_idx == 0 and tensor_idx < 2 and (src_idx - src_start) < 5) {
                        var debug_buf: [128]u8 = undefined;
                        const msg = std.fmt.bufPrint(debug_buf[0..], "[QCONCAT_DEBUG] Requant[t={d},idx={d}]: input={d}, dequant={}, scaled={}, clamped={}, output={d}\n", .{ tensor_idx, src_idx - src_start, input.data[src_idx], input_dequant, scaled_result, clamped_result, quantized_output }) catch "Debug format error\n";
                        debug_buf[msg.len] = 0;
                    }
                }
            }

            // Update the offset for the next copy
            offset += copy_size;
        }
    }

    // DEBUG: Final output analysis
    if (false) { // DISABLED
        var debug_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(debug_buf[0..], "[QCONCAT_DEBUG] Output analysis: size={d}\n", .{output.size}) catch "Debug format error\n";
        debug_buf[msg.len] = 0;

        // Check first 10 values
        for (0..@min(10, output.size)) |i| {
            const msg2 = std.fmt.bufPrint(debug_buf[0..], "  out[{d}] = {d}\n", .{ i, output.data[i] }) catch continue;
            debug_buf[msg2.len] = 0;
        }

        // Check diversity
        if (output.size > 1) {
            const first_val = output.data[0];
            var all_same = true;
            for (output.data[0..@min(100, output.size)]) |val| {
                if (val != first_val) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) {
                const msg3 = std.fmt.bufPrint(debug_buf[0..], "  WARNING: All values identical: {d}\n", .{first_val}) catch "";
                if (msg3.len > 0) {
                    debug_buf[msg3.len] = 0;
                }
            } else {}
        }
    }
}

/// Calculate output shape for QLinearConcat - same as regular concatenate
pub fn get_qlinearconcat_output_shape(
    input_shapes: []const []const usize,
    axis: isize,
) ![]usize {
    return concatenate.get_concatenate_output_shape(input_shapes, axis);
}
