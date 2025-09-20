const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

// Import existing average pool operation for logic and shape calculation
const averagepool = @import("op_averagePool.zig");

/// QLinearAveragePool operation following ONNX specification
/// Performs quantized average pooling using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(averagepool(dequantize(x)), y_scale, y_zero_point)
pub fn qlinearaveragepool(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    // AveragePool parameters
    kernel_shape: []const usize,
    strides: ?[]const usize,
    dilations: ?[]const usize,
    pads: ?[]const usize,
    auto_pad: averagepool.AutoPadType,
    ceil_mode: bool,
    count_include_pad: bool,
) !Tensor(InputType) {
    // Input validation
    if (x.shape.len < 3) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_scale.size != 1 or y_scale.size != 1) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_zero_point.size != 1 or y_zero_point.size != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output shape using existing averagepool logic
    const output_shape = try get_qlinearaveragepool_output_shape(
        x.shape,
        kernel_shape,
        strides orelse &[_]usize{1} ** kernel_shape.len,
        dilations orelse &[_]usize{1} ** kernel_shape.len,
        pads orelse &[_]usize{0} ** (kernel_shape.len * 2),
        auto_pad,
        ceil_mode,
    );
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized average pooling
    try lean_qlinearaveragepool(
        InputType,
        ScaleType,
        ZeroPointType,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        &output,
        kernel_shape,
        strides orelse &[_]usize{1} ** kernel_shape.len,
        dilations orelse &[_]usize{1} ** kernel_shape.len,
        pads orelse &[_]usize{0} ** (kernel_shape.len * 2),
        auto_pad,
        count_include_pad,
    );

    return output;
}

/// Lean version of QLinearAveragePool that operates on pre-allocated output tensor
pub fn lean_qlinearaveragepool(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    output: *Tensor(InputType),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: averagepool.AutoPadType,
    count_include_pad: bool,
) !void {
    const OutputType = @TypeOf(output.data[0]);

    // Skip computation for placeholder tensors
    if (x.size == 1 and output.size == 1 and x.shape.len == 1 and output.shape.len == 1) {
        output.data[0] = if (@typeInfo(OutputType) == .int) 0 else 0.0;
        return;
    }

    // Get quantization parameters
    const x_scale_val = x_scale.data[0];
    const x_zero_point_val = x_zero_point.data[0];
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = y_zero_point.data[0];

    // DEBUG: QLinearAveragePool parameters (disabled for build compatibility)

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

    const input_rank = x.shape.len;
    const spatial_dims = input_rank - 2;

    if (kernel_shape.len != spatial_dims) return TensorMathError.InvalidDimensions;
    if (strides.len != spatial_dims) return TensorMathError.InvalidDimensions;
    if (dilations.len != spatial_dims) return TensorMathError.InvalidDimensions;

    const batch_size = x.shape[0];
    const channels = x.shape[1];

    // Calculate effective padding
    var effective_pads = try pkg_allocator.alloc(isize, spatial_dims * 2);
    defer pkg_allocator.free(effective_pads);

    switch (auto_pad) {
        .NOTSET => {
            for (0..spatial_dims) |i| {
                effective_pads[i] = if (i < pads.len) @as(isize, @intCast(pads[i])) else 0;
                effective_pads[i + spatial_dims] = if (i + spatial_dims < pads.len) @as(isize, @intCast(pads[i + spatial_dims])) else 0;
            }
        },
        .VALID => {
            @memset(effective_pads, 0);
        },
        .SAME_UPPER, .SAME_LOWER => {
            for (0..spatial_dims) |i| {
                const input_size = x.shape[i + 2];
                const output_size = output.shape[i + 2];
                const effective_kernel = (kernel_shape[i] - 1) * dilations[i] + 1;
                const total_pad = @max(0, @as(isize, @intCast((output_size - 1) * strides[i] + effective_kernel)) - @as(isize, @intCast(input_size)));

                if (auto_pad == .SAME_UPPER) {
                    effective_pads[i] = total_pad / 2;
                    effective_pads[i + spatial_dims] = total_pad - effective_pads[i];
                } else {
                    effective_pads[i + spatial_dims] = total_pad / 2;
                    effective_pads[i] = total_pad - effective_pads[i + spatial_dims];
                }
            }
        },
    }

    // Perform pooling operation
    for (0..batch_size) |b| {
        for (0..channels) |c| {
            // Calculate output indices for this batch and channel
            var output_spatial_indices = try pkg_allocator.alloc(usize, spatial_dims);
            defer pkg_allocator.free(output_spatial_indices);
            @memset(output_spatial_indices, 0);

            var done = false;
            while (!done) {
                // Calculate pooling window
                var sum: f64 = 0.0;
                var count: usize = 0;

                // Iterate through kernel
                var kernel_indices = try pkg_allocator.alloc(usize, spatial_dims);
                defer pkg_allocator.free(kernel_indices);
                @memset(kernel_indices, 0);

                var kernel_done = false;
                while (!kernel_done) {
                    // Calculate input position
                    var valid_input = true;
                    var input_spatial_indices = try pkg_allocator.alloc(isize, spatial_dims);
                    defer pkg_allocator.free(input_spatial_indices);

                    for (0..spatial_dims) |i| {
                        const output_pos = @as(isize, @intCast(output_spatial_indices[i]));
                        const kernel_pos = @as(isize, @intCast(kernel_indices[i]));
                        const stride = @as(isize, @intCast(strides[i]));
                        const dilation = @as(isize, @intCast(dilations[i]));

                        input_spatial_indices[i] = output_pos * stride + kernel_pos * dilation - effective_pads[i];

                        if (input_spatial_indices[i] < 0 or input_spatial_indices[i] >= @as(isize, @intCast(x.shape[i + 2]))) {
                            valid_input = false;
                            if (!count_include_pad) break;
                        }
                    }

                    if (valid_input) {
                        // Calculate linear input index
                        var input_idx: usize = b * channels;
                        for (0..channels) |_| break;
                        input_idx = (input_idx + c);
                        for (0..spatial_dims) |i| {
                            input_idx *= x.shape[i + 2];
                            input_idx += @as(usize, @intCast(input_spatial_indices[i]));
                        }

                        // Dequantize input value and add to sum
                        const input_val = x.data[input_idx];
                        const dequant_val =
                            (asF32(InputType, input_val) - asF32(ZeroPointType, x_zero_point_val)) * asF32(ScaleType, x_scale_val);

                        sum += @as(f64, dequant_val);
                        count += 1;
                    } else if (count_include_pad) {
                        count += 1;
                    }

                    // Increment kernel indices
                    var kernel_carry = true;
                    var k_idx = spatial_dims;
                    while (k_idx > 0 and kernel_carry) {
                        k_idx -= 1;
                        kernel_indices[k_idx] += 1;
                        if (kernel_indices[k_idx] < kernel_shape[k_idx]) {
                            kernel_carry = false;
                        } else {
                            kernel_indices[k_idx] = 0;
                        }
                    }
                    kernel_done = kernel_carry;
                }

                // Calculate average and quantize
                const avg_float = if (count > 0) @as(f32, @floatCast(sum)) / @as(f32, @floatFromInt(count)) else 0.0;
                const scaled_result = (avg_float / asF32(ScaleType, y_scale_val)) + asF32(ZeroPointType, y_zero_point_val);

                // Clamp to valid range for output type
                // For QLinear operations, output is always quantized (integer type)
                const min_val = @as(f32, @floatFromInt(std.math.minInt(OutputType)));
                const max_val = @as(f32, @floatFromInt(std.math.maxInt(OutputType)));
                const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

                // Calculate output index and store result
                var output_idx: usize = b * channels;
                for (0..channels) |_| break;
                output_idx = (output_idx + c);
                for (0..spatial_dims) |i| {
                    output_idx *= output.shape[i + 2];
                    output_idx += output_spatial_indices[i];
                }

                // For QLinear operations, output is always quantized (integer type)
                const quantized_output = @as(OutputType, @intFromFloat(@round(clamped_result)));
                output.data[output_idx] = quantized_output;

                // DEBUG: First few pooling operations
                if (false) { // DISABLED
                    if (b == 0 and c < 2 and output_spatial_indices[0] < 3) {
                        var debug_buf: [128]u8 = undefined;
                        const msg = std.fmt.bufPrint(debug_buf[0..], "[QPOOL_DEBUG] Pool[b={d},c={d},pos=[{d},{d}]]: avg_float={}, scaled={}, clamped={}, quantized={d}\n", .{ b, c, output_spatial_indices[0], if (spatial_dims > 1) output_spatial_indices[1] else 0, avg_float, scaled_result, clamped_result, quantized_output }) catch "Debug format error\n";
                        debug_buf[msg.len] = 0;
                    }
                }

                // Increment output spatial indices
                var carry = true;
                var idx = spatial_dims;
                while (idx > 0 and carry) {
                    idx -= 1;
                    output_spatial_indices[idx] += 1;
                    if (output_spatial_indices[idx] < output.shape[idx + 2]) {
                        carry = false;
                    } else {
                        output_spatial_indices[idx] = 0;
                    }
                }
                done = carry;
            }
        }
    }

    // DEBUG: Final output analysis
    if (false) { // DISABLED
        var debug_buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(debug_buf[0..], "[QPOOL_DEBUG] Output analysis: size={d}\n", .{output.size}) catch "Debug format error\n";
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

/// Calculate output shape for QLinearAveragePool - same as regular AveragePool
pub fn get_qlinearaveragepool_output_shape(
    input_shape: []const usize,
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: averagepool.AutoPadType,
    ceil_mode: bool,
) ![]usize {
    return averagepool.get_onnx_averagepool_output_shape(
        input_shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
}
