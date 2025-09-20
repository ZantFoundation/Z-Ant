const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const tensorType = zant.core.tensor.TensorType;
const tensorDetails = zant.core.tensor.TensorDetails;
const quantDetails = zant.core.tensor.QuantDetails;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const quantMath = zant.core.tensor.quantized_math;
const mat_mul = quantMath.quant_mat_mul;
const blocked_mat_mul = quantMath.quant_blocked_mat_mul;
const addPaddingAndDilation = quantMath.addPaddingAndDilation;
const op_transpose = @import("../TensorMath/lib_shape_math/op_transpose.zig");

// CONVOLVE -----------------------------------------------------------------------------------------------------------------------

/// Multidim Conv
/// INPUT:
///     INPUT[input.shape.len - 4] -> batches
///     INPUT[input.shape.len - 3] -> input channels
///     INPUT[input.shape.len - 2] -> rows
///     INPUT[input.shape.len - 1] -> cols
/// KERNEL:
///     KERNEL[kernel.shape.len - 4] -> filters
///     KERNEL[kernel.shape.len - 3] -> channels
///     KERNEL[kernel.shape.len - 2] -> rows
///     KERNEL[kernel.shape.len - 1] -> cols
/// OUTPUT:
///     OUTPUT[output.shape.len - 4] -> input_batch
///     OUTPUT[output.shape.len - 3] -> output channels (number_of_kernel_filters)
///     OUTPUT[output.shape.len - 2] -> rows
///     OUTPUT[output.shape.len - 1] -> cols
/// Convolution tensor with bias
/// TODO: create 2d convolution, atm is 3 or more dimensions
/// TODO: add better check on output size wrt input and kernel
///
///
///
pub var log_functionC: ?*const fn ([*c]u8) callconv(.C) void = null;

pub export fn setQuantLogFunctionC(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_functionC = func;
}

/// OnnxConv
pub fn OnnxConv(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !Tensor(T) {
    // Input validation
    if (input.shape.len != 4 or kernel.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }
    // Quantized inputs validation
    if (input.details != .quant or kernel.details != .quant or (if (bias) bias.details != .quant))
        return TensorError.NotQuantizedTensor;

    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = kernel.shape[0];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    // Group validation (minimal here, more detailed in OnnxConvLean)
    const actual_group = group orelse 1;
    const in_channels = input.shape[1];
    const kernel_channels_per_group = kernel.shape[1];
    if (in_channels % actual_group != 0 or out_channels % actual_group != 0 or kernel_channels_per_group != in_channels / actual_group) {
        // Basic check, detailed error handled by OnnxConvLean
        return TensorMathError.InvalidDimensions;
    }

    // Set default values for stride and dilation
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride_h;
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;

    // Calculate dilated kernel dimensions
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    // Calculate padding and output dimensions
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;
    var expected_out_height: usize = undefined;
    var expected_out_width: usize = undefined;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            expected_out_height = (in_height - dilated_kernel_h) / stride_h + 1;
            expected_out_width = (in_width - dilated_kernel_w) / stride_w + 1;
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            expected_out_height = ceilDiv(in_height, stride_h);
            expected_out_width = ceilDiv(in_width, stride_w);

            const total_pad_h = @max((expected_out_height - 1) * stride_h + dilated_kernel_h - in_height, 0);
            const total_pad_w = @max((expected_out_width - 1) * stride_w + dilated_kernel_w - in_width, 0);

            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = total_pad_h / 2;
                pad_h_end = total_pad_h - pad_h_begin;
                pad_w_begin = total_pad_w / 2;
                pad_w_end = total_pad_w - pad_w_begin;
            } else {
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;
                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        } else if (std.mem.eql(u8, pad_mode, "NOTSET")) {
            if (pads) |p| {
                if (p.len >= 4) {
                    pad_h_begin = p[0];
                    pad_w_begin = p[1];
                    pad_h_end = p[2];
                    pad_w_end = p[3];
                    expected_out_height = (in_height + pad_h_begin + pad_h_end - dilated_kernel_h) / stride_h + 1;
                    expected_out_width = (in_width + pad_w_begin + pad_w_end - dilated_kernel_w) / stride_w + 1;
                } else {
                    return TensorMathError.InvalidPadding;
                }
            } else {
                expected_out_height = (in_height - dilated_kernel_h) / stride_h + 1;
                expected_out_width = (in_width - dilated_kernel_w) / stride_w + 1;
            }
        } else {
            return TensorMathError.InvalidPadding;
        }
    } else {
        // No auto_pad, use explicit padding if provided
        const total_pad_h = if (pads) |p| if (p.len >= 4) p[0] + p[2] else 0 else 0;
        const total_pad_w = if (pads) |p| if (p.len >= 4) p[1] + p[3] else 0 else 0;
        expected_out_height = (in_height + total_pad_h - dilated_kernel_h) / stride_h + 1;
        expected_out_width = (in_width + total_pad_w - dilated_kernel_w) / stride_w + 1;
    }

    // Create output tensor with correct shape
    var output_shape = [_]usize{ input.shape[0], out_channels, expected_out_height, expected_out_width };
    var output = Tensor(T).fromShape(&pkg_allocator, &output_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConv: Failed to allocate output tensor")));
        }
        @panic("Memory allocation failed for output tensor in OnnxConv");
    };
    errdefer output.deinit();
    // Details propagation
    output.details = input.details;

    // Call the lean version
    try OnnxConvLean(T, input, kernel, &output, bias, stride, pads, dilations, actual_group, auto_pad);

    return output;
}

pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void {

    // Input validation: Ensure 4D tensors for input (N,C,H,W) and kernel (M,C/g,H,W)

    // --- Handle 3D input shape by assuming batch size = 1 ---
    var actual_input_shape: [4]usize = undefined;
    var input_ptr = input; // Use original pointer unless adjusted
    var temp_input_tensor: ?Tensor(T) = null; // Optional tensor for adjusted shape

    if (input.shape.len == 3) {
        // Assume batch size is 1 if input is 3D (C, H, W)
        actual_input_shape[0] = 1;
        actual_input_shape[1] = input.shape[0]; // Channels
        actual_input_shape[2] = input.shape[1]; // Height
        actual_input_shape[3] = input.shape[2]; // Width

        // Create a temporary tensor using fromArray - this allocates and copies
        const temp_tensor = try Tensor(T).fromArray(&pkg_allocator, input.data, &actual_input_shape);
        temp_input_tensor = temp_tensor; // Assign to the optional field
        input_ptr = &temp_input_tensor.?; // Point to the temporary tensor

    } else if (input.shape.len == 4) {
        @memcpy(&actual_input_shape, input.shape[0..4]);
    } else {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLeanError: Invalid input dimensions")));
        }
        return TensorMathError.InvalidDimensions;
    }
    // Defer deinitialization if a temporary tensor was created
    // Note: deinit on a fromArray tensor *will* free the copied data/shape
    defer if (temp_input_tensor) |*t| t.deinit(); // Deinit only if temp_input_tensor is not null

    if (kernel.shape.len != 4) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLeanError: Invalid kernel dimensions")));
        }
        return TensorMathError.InvalidDimensions;
    }
    // --- End Shape Adjustment ---

    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }

    // Use actual_input_shape for calculations
    const in_height = actual_input_shape[2];
    const in_width = actual_input_shape[3];
    const out_channels = kernel.shape[0];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];
    const kernel_channels_per_group = kernel.shape[1]; // ONNX Kernel shape [M, C/g, kH, kW]
    const in_channels = actual_input_shape[1]; // C
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }
    // Group validation
    const actual_group = group orelse 1;

    if (in_channels % actual_group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLeanError: Input channels not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    if (out_channels % actual_group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLeanError: Output channels not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    // Check if kernel's channel dimension (C/g) matches input channels / group
    if (kernel_channels_per_group != in_channels / actual_group) {
        if (log_functionC) |log_func| {
            var buf: [256]u8 = undefined;
            _ = std.fmt.bufPrint(&buf, "OnnxConvLeanError: Kernel channels mismatch. Kernel C/g={}, Input C/g={d}/{d}={d}", .{ kernel_channels_per_group, in_channels, actual_group, in_channels / actual_group }) catch unreachable;
            log_func(@constCast(@ptrCast(&buf)));
        }
        return TensorMathError.InvalidDimensions;
    }

    // Set default values for stride and dilation
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride[0];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;

    // Calculate dilated kernel dimensions
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    // Padding variables
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }
    // Handle padding: Either auto_pad or explicit pads
    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            // No padding
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            // Calculate output dimensions for SAME padding
            const out_height = ceilDiv(in_height, stride_h);
            const out_width = ceilDiv(in_width, stride_w);

            // Calculate total padding required
            const total_pad_h = @max((out_height - 1) * stride_h + dilated_kernel_h - in_height, 0);
            const total_pad_w = @max((out_width - 1) * stride_w + dilated_kernel_w - in_width, 0);

            // Distribute padding based on SAME_UPPER or SAME_LOWER
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
        } else if (std.mem.eql(u8, pad_mode, "NOTSET")) {
            if (pads) |p| {
                // Use explicit padding if provided
                if (p.len >= 4) {
                    pad_h_begin = p[0];
                    pad_w_begin = p[1];
                    pad_h_end = p[2];
                    pad_w_end = p[3];
                } else if (p.len == 2) { // Handle symmetric padding [h, w]
                    pad_h_begin = p[0];
                    pad_h_end = p[0];
                    pad_w_begin = p[1];
                    pad_w_end = p[1];
                } else {
                    // Default to zero padding if pads length is not 2 or >= 4
                    pad_h_begin = 0;
                    pad_w_begin = 0;
                    pad_h_end = 0;
                    pad_w_end = 0;
                }
            } else {
                // pads is null, default to zero padding
                pad_h_begin = 0;
                pad_h_end = 0;
                pad_w_begin = 0;
                pad_w_end = 0;
                // No error needed here anymore
                // if (log_functionC) |log_func| {
                //     log_func(@constCast(@ptrCast("OnnxConvLeanError2")));
                // }
                // return TensorMathError.InvalidPadding;
            }
        } else if (!std.mem.eql(u8, pad_mode, "NOTSET")) {} else if (!std.mem.eql(u8, pad_mode, "NOTSET")) {
            return TensorMathError.InvalidPadding;
        }
    }
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }
    // Create padded input tensor
    var padded_shape = [_]usize{
        actual_input_shape[0], // Use adjusted batch size
        actual_input_shape[1], // Use adjusted channels
        actual_input_shape[2] + pad_h_begin + pad_h_end,
        actual_input_shape[3] + pad_w_begin + pad_w_end,
    };
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }
    var padded_input = Tensor(T).fromShape(&pkg_allocator, &padded_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLean: Failed to allocate padded_input tensor")));
        }
        @panic("Memory allocation failed for padded_input tensor");
    };
    defer padded_input.deinit();
    try padded_input.set(0, 0); // Initialize to zeros
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean1")));
    }
    // Copy input data into the padded tensor, offset by beginning padding
    // Use input_ptr which points to either the original input or the temp view
    for (0..input_ptr.shape[0]) |b| { // Iterate using the shape of the tensor pointed to by input_ptr
        for (0..input_ptr.shape[1]) |c| {
            for (0..input_ptr.shape[2]) |h| {
                for (0..input_ptr.shape[3]) |w| {
                    const val = try input_ptr.get_at(&[_]usize{ b, c, h, w }); // Get from correct tensor
                    try padded_input.set_at(&[_]usize{ b, c, h + pad_h_begin, w + pad_w_begin }, val);
                }
            }
        }
    }

    // Handle bias: Use zero bias if none provided
    var zero_bias: Tensor(T) = undefined;
    if (bias == null) {
        var bias_shape = [_]usize{out_channels};
        zero_bias = Tensor(T).fromShape(&pkg_allocator, &bias_shape) catch {
            if (log_functionC) |log_func| {
                log_func(@constCast(@ptrCast("OnnxConvLean: Failed to allocate zero_bias tensor")));
            }
            @panic("Memory allocation failed for zero_bias tensor");
        };
        errdefer zero_bias.deinit();
        try zero_bias.set(0, 0);
    }

    // Prepare stride and dilation arrays for convolution
    var stride_arr = [_]usize{ stride_h, stride_w };
    var dilation_arr = [_]usize{ dilation_h, dilation_w };
    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean2")));
    }
    // Perform convolution with padded input, kernel, and bias
    var result = convolve_tensor_with_bias(
        T,
        &padded_input,
        kernel,
        if (bias) |b| b else &zero_bias,
        &stride_arr,
        &dilation_arr,
        actual_group, // Pass the group parameter
    ) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLean: Failed in convolve_tensor_with_bias")));
        }
        @panic("Failed in convolve_tensor_with_bias");
    };
    defer result.deinit();

    // Clean up zero bias if it was created
    if (bias == null) {
        zero_bias.deinit();
    }

    //print  result e output shape

    // Validate output dimensions and copy result
    // Output shape should match the expected 4D shape based on actual_input_shape
    var shape_match = false;
    if (output.shape.len == 4) {
        // Compare all 4 dimensions if output is 4D
        shape_match = std.mem.eql(usize, result.shape[0..4], output.shape[0..4]);
    } else if (output.shape.len == 3) {
        // Compare dimensions C, H, W if output is 3D
        // Result is always 4D [B=1, C, H, W] from convolve_tensor_with_bias
        shape_match = std.mem.eql(usize, result.shape[1..4], output.shape[0..3]);
    } // else: Let the data length check catch other invalid output shapes

    if (!shape_match) {
        // Adjust error message or logic if needed, considering the 3D input case
        if (log_functionC) |log_func| {
            var buf: [256]u8 = undefined;
            _ = std.fmt.bufPrint(&buf, "OnnxConvLeanError: Output shape mismatch. Result: {any}, Expected: {any}", .{ result.shape, output.shape }) catch unreachable;
            log_func(@constCast(@ptrCast(&buf)));
        }
        return TensorMathError.InvalidDimensions;
    }
    if (result.data.len != output.data.len) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("OnnxConvLeanError: Output data length mismatch")));
        }
        return TensorMathError.InvalidDimensions;
    }
    @memcpy(output.data, result.data);
}

fn ceilDiv(a: usize, b: usize) usize {
    if (b == 0) {
        @panic("Division by zero");
    }
    return (a + b - 1) / b;
}

pub fn convolve_tensor_with_bias(
    comptime T: type,
    comptime T1: type, // Bias data type should be int32
    input: *Tensor(T),
    kernel: *Tensor(T),
    bias: ?*Tensor(T1),
    stride: []const usize,
    dilations: ?[]const usize,
    group: usize,
) !Tensor(T) {
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;
    const dilated_kernel_h = (kernel.shape[2] - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel.shape[3] - 1) * dilation_w + 1;

    const batch_size = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const num_filters = kernel.shape[0]; // Total output channels (M)
    const kernel_channels_per_group = kernel.shape[1]; // Input channels per group (C/g)
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    // Quantized data type max and min
    const max_value = if (@typeInfo(T) == .int) std.math.maxInt(T) else 0;
    const min_value = if (@typeInfo(T) == .int) std.math.minInt(T) else 0;

    // --- Group Validations ---
    if (in_channels % group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("Error: Input channels not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    if (num_filters % group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("Error: Output channels (num_filters) not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    if (kernel_channels_per_group != in_channels / group) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("Error: Kernel channels per group mismatch")));
        }
        return TensorMathError.InvalidDimensions; // Kernel's channel dim should match input channels / group
    }
    const channels_per_group = in_channels / group; // C/g
    const filters_per_group = num_filters / group; // M/g
    // --- End Group Validations ---

    // Special case handling for tiny inputs with larger or equal-sized kernels with stride > 1
    if ((in_height <= dilated_kernel_h or in_width <= dilated_kernel_w) or
        (stride.len > 0 and stride[0] > 1 and in_height == dilated_kernel_h) or
        (stride.len > 1 and stride[1] > 1 and in_width == dilated_kernel_w) or
        (in_height == 1 and in_width == 1))
    {
        var zero: i32 = undefined;
        var scale: f32 = undefined;
        // For a small input with a larger kernel or equal kernel with stride>1,
        // we'll return a tensor with shape [batch_size, num_filters, 1, 1]
        var output_shape = [_]usize{ batch_size, num_filters, 1, 1 };
        var output = Tensor(T).fromShape(&pkg_allocator, &output_shape) catch {
            if (log_functionC) |log_func| {
                log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed to allocate output for special case")));
            }
            @panic("Memory allocation failed for output tensor");
        };
        errdefer output.deinit();
        // // Details propagation
        // output.details = input.details;

        switch (input.details) {
            .quant => {
                const input_zero_point = @as(i32, @intCast(try input.get_zero_point()));
                const kernel_zero_point = @as(i32, @intCast(try kernel.get_zero_point()));

                scale *= try input.get_scale_factor() * try kernel.get_scale_factor();
                var effective_scale: f32 = try input.get_scale_factor() * try kernel.get_scale_factor() / scale;
                // shift_correction and effective_scale normalization in [1, 0.5) range
                var shift_correction: u5 = 0;
                while (effective_scale >= 1) {
                    effective_scale /= 2;
                    shift_correction += 1;
                }
                while (effective_scale < 0.5) {
                    effective_scale *= 2;
                    shift_correction -= 1;
                }
                // multiplier
                const shift: u5 = 31;
                const multiplier: i32 = @intFromFloat(@round(effective_scale * @as(f32, 1 << shift)));
                // zero_point
                zero = 0; // Output zero point set at zero by default

                // For each batch and filter
                for (0..batch_size) |b| {
                    for (0..num_filters) |f| {
                        var sum: i32 = 0; // i32 accumulation
                        const current_group = f / filters_per_group;
                        //const kernel_channel_start = 0; // Kernel channels dim is already C/g
                        //const kernel_channel_end = kernel_channels_per_group;
                        const input_channel_start = current_group * channels_per_group;
                        const input_channel_end = input_channel_start + channels_per_group;

                        // For a small input, we need to check if kernel dimensions are valid
                        if (kernel_channels_per_group >= 1 and
                            kernel_height >= 1 and
                            kernel_width >= 1)
                        {
                            // Center of the kernel for 3x3 or other odd-sized kernels
                            const center_h = kernel_height / 2;
                            const center_w = kernel_width / 2;

                            // For each input channel (limited to match kernel.shape[1])
                            // Iterate over relevant input channels for this group
                            for (input_channel_start..input_channel_end) |in_c| {
                                // Map input channel index 'in_c' to the kernel's channel index 'k_c' (0 to C/g - 1)
                                const k_c = in_c - input_channel_start;
                                // Ensure we don't go past the actual input channels if padding happened weirdly
                                if (in_c >= input.shape[1]) continue;
                                // Ensure kernel channel index is valid (should be due to checks, but safety)
                                if (k_c >= kernel_channels_per_group) continue;

                                const input_val = try input.get_at(&[_]usize{ b, in_c, 0, 0 });
                                const kernel_val = try kernel.get_at(&[_]usize{ f, k_c, center_h, center_w });

                                sum += (@as(i32, input_val) - input_zero_point) * (@as(i32, kernel_val) - kernel_zero_point);
                            }
                        }

                        // Add bias if provided
                        const bias_val = if (bias) |bias_tensor| try bias_tensor.get_at(&[_]usize{f}) else 0;

                        // Clamping
                        const accumulated_sum: isize = bias_val + zero + ((sum * multiplier) >> (shift + shift_correction));
                        if (accumulated_sum > max_value) {
                            sum = max_value;
                        } else if (accumulated_sum < min_value) {
                            sum = min_value;
                        } else {
                            sum = @as(i32, @intCast(accumulated_sum));
                        }

                        try output.set_at(&[_]usize{ b, f, 0, 0 }, @as(T, @intCast(sum)));
                    }
                }
            },
            else => return TensorError.NotQuantizedTensor,
        }

        output.details = tensorDetails{
            .quant = quantDetails{
                .tensorType = tensorType.QuantTensor,
                .scale_factor = scale,
                .zero_point = zero,
            },
        };

        return output;
    }

    // Calculate output dimensions (standard case)

    const out_height: usize = @divFloor(in_height - dilated_kernel_h, stride[0]) + 1;
    const out_width: usize = @divFloor(in_width - dilated_kernel_w, stride[1]) + 1;

    if (out_height <= 0 or out_width <= 0) {
        return TensorMathError.InvalidDimensions;
    }

    var output_shape = [_]usize{ batch_size, num_filters, out_height, out_width };
    var output = Tensor(T).fromShape(&pkg_allocator, &output_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed to allocate output tensor")));
        }
        @panic("Memory allocation failed for output tensor in convolve_tensor_with_bias");
    };
    errdefer output.deinit();
    // Details propagation
    output.details = input.details;

    const kernel_size = [2]usize{ kernel_height, kernel_width };
    const stride_size = [2]usize{ stride[0], stride[1] };

    // Pass group parameter to im2col
    var input_col = im2col(T, input, kernel_size, stride_size, dilations, group) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed in im2col")));
        }
        @panic("Failed in im2col");
    };
    defer input_col.deinit();

    if (log_functionC) |log_func| {
        log_func(@constCast(@ptrCast("OnnxConvLean3")));
    }
    // kernel_elements_per_filter = (C/g) * kH * kW
    const kernel_elements_per_filter = kernel_channels_per_group * kernel_height * kernel_width;

    // Print dimensions to debug
    if (log_functionC) |log_func| {
        var buf: [128]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "input_col shape: {any}, kernel elements per filter: {}", .{ input_col.shape, kernel_elements_per_filter }) catch "";
        log_func(@constCast(@ptrCast(&buf)));
    }

    // Reshape kernel: [M, C/g, kH, kW] -> [M, K] where K = C/g * kH * kW
    var kernel_matrix_shape = [_]usize{ num_filters, kernel_elements_per_filter };
    var kernel_matrix = Tensor(T).fromShape(&pkg_allocator, &kernel_matrix_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed to allocate kernel_matrix tensor")));
        }
        @panic("Memory allocation failed for kernel_matrix tensor");
    };
    defer kernel_matrix.deinit();

    // Reshape kernel data [M, C/g * kH * kW]
    for (0..num_filters) |f| {
        const kernel_filter_offset = f * kernel_elements_per_filter;
        for (0..kernel_elements_per_filter) |k_idx| {
            // Kernel data is already contiguous for [C/g, kH, kW] within each filter 'f'
            try kernel_matrix.set_at(&[_]usize{ f, k_idx }, kernel.data[kernel_filter_offset + k_idx]);
        }
    }

    // Perform group-wise matrix multiplication logic OR standard mat_mul
    // input_col (I) shape: [N, group * K], where N = B*OH*OW, K = C/g * kH * kW
    // kernel_matrix (W) shape: [M, K]
    // Output (O) shape: [N, M]
    const N = input_col.shape[0]; // B * OH * OW
    const M = num_filters; // Total output filters
    const K = kernel_elements_per_filter; // C/g * kH * kW

    // Declare result outside the branches
    var result: ?Tensor(T) = null; // Initialize as null
    // Use a sentinel value check for defer, assuming .data.ptr is null before assignment
    // If allocation fails inside branches, it won't be deferred. If it succeeds, it will be.
    defer if (result) |*r| r.deinit(); // Deinit if result has a value

    if (group == 1) {
        // --- Standard Convolution (group = 1) ---

        // Transpose the kernel matrix: [M, K] -> [K, M]
        var kernel_matrix_transposed = try op_transpose.transposeLastTwo(T, &kernel_matrix);
        defer kernel_matrix_transposed.deinit();

        // Check cache size to potentially use blocked mat mul
        const vals_in_cache = std.atomic.cache_line / @sizeOf(T);
        // Multiply input_col [N, K] by kernel_matrix_transposed [K, M]
        if (kernel_matrix_transposed.shape[kernel_matrix_transposed.shape.len - 1] > vals_in_cache) { // Check transposed shape
            result = blocked_mat_mul(T, &input_col, &kernel_matrix_transposed) catch {
                if (log_functionC) |log_func| {
                    log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed in blocked_mat_mul (group=1)")));
                }
                @panic("Failed in blocked mat_mul (group=1)");
            };
        } else {
            result = mat_mul(T, &input_col, &kernel_matrix_transposed) catch {
                if (log_functionC) |log_func| {
                    log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed in mat_mul (group=1)")));
                }
                @panic("Failed in mat_mul (group=1)");
            };
        }
    } else {
        // --- Grouped Convolution (group > 1) ---
        // Allocate result tensor for the manual loop
        var result_shape = [_]usize{ N, M };
        result = Tensor(T).fromShape(&pkg_allocator, &result_shape) catch {
            if (log_functionC) |log_func| {
                log_func(@constCast(@ptrCast("convolve_tensor_with_bias: Failed to allocate result tensor for matmul (group>1)")));
            }
            @panic("Memory allocation failed for result tensor (group>1)");
        };
        // Initialize result tensor
        try result.?.set(0, 0); // Use .? to unwrap the optional

        var zero: i32 = undefined;
        var scale: f32 = undefined;
        // Optimized loops for group-wise dot product calculation
        var n: usize = 0;
        switch (output.details) {
            .quant => {
                const input_zero_point = @as(i32, @intCast(try input.get_zero_point()));
                const kernel_zero_point = @as(i32, @intCast(try kernel.get_zero_point()));

                scale = try input.get_scale_factor() * try kernel.get_scale_factor();
                var effective_scale: f32 = try input.get_scale_factor() * try kernel.get_scale_factor() / scale;
                // shift_correction and effective_scale normalization in [1, 0.5] range
                var shift_correction: u5 = 0;
                while (effective_scale >= 1) {
                    effective_scale /= 2;
                    shift_correction -= 1;
                }
                while (effective_scale < 0.5) {
                    effective_scale *= 2;
                    shift_correction += 1;
                }
                // multiplier
                const shift: u5 = 31;
                const multiplier: i32 = @intFromFloat(@round(effective_scale * @as(f32, 1 << shift)));
                // zero_point
                zero = 0; // Output zero point set at zero by default

                while (n < N) : (n += 1) {
                    const input_col_row_offset = n * input_col.shape[1]; // Offset for start of row n in input_col.data

                    var m: usize = 0;
                    while (m < M) : (m += 1) {
                        const g = m / filters_per_group; // Determine the group for the current filter m
                        const kernel_filter_offset = m * K; // Offset for start of filter m in kernel_matrix.data
                        // Calculate offset into input_col for the specific group g's data for this row n
                        const input_col_group_offset = input_col_row_offset + g * K;

                        // Calculate dot product for O[n, m] using data from the correct group
                        var dot_product: i32 = 0;
                        var k: usize = 0;
                        while (k < K) : (k += 1) {
                            // Accessing data directly via pointers and offsets
                            const input_data = input_col.data[input_col_group_offset + k];
                            const kernel_data = kernel_matrix.data[kernel_filter_offset + k];
                            dot_product += (@as(i32, input_data) - input_zero_point) * (@as(i32, kernel_data) - kernel_zero_point);
                        }

                        // Clamping
                        const accumulated_prod: isize = zero + ((dot_product * multiplier) >> (shift + shift_correction));
                        dot_product = if (accumulated_prod > max_value) {
                            max_value;
                        } else if (accumulated_prod < min_value) {
                            min_value;
                        } else {
                            @as(i32, @intCast(accumulated_prod));
                        };

                        // Store the result for this output element
                        result.?.data[n * M + m] = @as(T, @intCast(dot_product)); // Use ?.data to access optional
                    }
                }
                output.details = tensorDetails{
                    .quant = quantDetails{
                        .tensorType = tensorType.QuantTensor,
                        .scale_factor = scale,
                        .zero_point = zero,
                    },
                };
            },
            else => return TensorError.NotQuantizedTensor,
        }
    } // End if (group == 1) else

    // --- Post-computation checks and reshaping ---
    // The 'result' tensor is now populated correctly based on the group value

    // Reshape result [N, M] -> [B, M, OH, OW] and add bias
    //var idx: usize = 0;
    for (0..batch_size) |b| {
        for (0..num_filters) |f| { // Iterate through filters first for better cache locality potentially
            const bias_val = if (bias) |bias_tensor| try bias_tensor.get_at(&[_]usize{f}) else 0;
            for (0..out_height) |h| {
                for (0..out_width) |w| {
                    // Calculate index in the flat result tensor [N, M]
                    const n_idx = b * (out_height * out_width) + h * out_width + w;
                    const val = result.?.data[n_idx * num_filters + f]; // Access result.?.data[n_idx, f]
                    // Set value in the final output tensor [B, M, OH, OW]
                    try output.set_at(&[_]usize{ b, f, h, w }, val + bias_val);
                }
            }
        }
    }

    return output;
}

/// Memory-efficient convolution implementation without using im2col.
/// Optimized for speed using pointer arithmetic and SIMD.
pub fn convolve_tensor_with_bias_memory_efficient(
    comptime T: type,
    input: *Tensor(T),
    kernel: *const Tensor(T),
    bias: ?*const Tensor(T),
    stride: []const usize,
    dilations: ?[]const usize,
    group: usize, // Added group parameter
) !Tensor(T) {
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;
    const dilated_kernel_h = (kernel.shape[2] - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel.shape[3] - 1) * dilation_w + 1;

    const batch_size = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];

    const num_filters = kernel.shape[0]; // M (total output channels)
    const kernel_channels_per_group = kernel.shape[1]; // C/g
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    // Quantized data type max and min
    var max_value = undefined;
    var min_value = undefined;
    if (@typeInfo(T) == .int) {
        max_value = std.math.maxInt(T);
        min_value = std.math.minInt(T);
    }

    // --- Group Validations ---
    if (in_channels % group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_mem_eff Error: Input channels not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    if (num_filters % group != 0) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_mem_eff Error: Output channels (num_filters) not divisible by group")));
        }
        return TensorMathError.InvalidGroupParameter;
    }
    if (kernel_channels_per_group != in_channels / group) {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_mem_eff Error: Kernel channels per group mismatch")));
        }
        return TensorMathError.InvalidDimensions; // Kernel's channel dim should match input channels / group
    }
    const channels_per_group = in_channels / group; // C/g
    const filters_per_group = num_filters / group; // M/g
    // --- End Group Validations ---

    // --- Special case handling for tiny inputs ---
    if ((in_height <= dilated_kernel_h or in_width <= dilated_kernel_w) or
        (stride.len > 0 and stride[0] > 1 and in_height == dilated_kernel_h) or
        (stride.len > 1 and stride[1] > 1 and in_width == dilated_kernel_w) or
        (in_height == 1 and in_width == 1))
    {
        var output_shape = [_]usize{ batch_size, num_filters, 1, 1 };
        var output = Tensor(T).fromShape(&pkg_allocator, &output_shape) catch {
            if (log_functionC) |log_func| {
                log_func(@constCast(@ptrCast("convolve_mem_eff: Failed allocate output special case")));
            }
            @panic("Memory allocation failed for output tensor (special case)");
        };
        errdefer output.deinit();
        // Details propagation
        output.details = input.details;
        switch (output.details) {
            .quant => |*qd| {
                const input_zero_point = @as(i32, @intCast(try input.get_zero_point()));
                const kernel_zero_point = @as(i32, @intCast(try kernel.get_zero_point()));

                qd.scale_factor = try input.get_scale_factor() * try kernel.get_scale_factor();
                var effective_scale: f32 = try input.get_scale_factor() * try kernel.get_scale_factor() / qd.scale_factor;
                // shift_correction and effective_scale normalization in [1, 0.5] range
                var shift_correction: usize = 0;
                while (effective_scale > 1) {
                    effective_scale /= 2;
                    shift_correction += 1;
                }
                while (effective_scale < 0.5) {
                    effective_scale *= 2;
                    shift_correction -= 1;
                }
                // multiplier
                const shift = 31;
                const multiplier: i32 = @intFromFloat(@round(effective_scale * @as(f32, 1 << shift)));
                // zero_point
                qd.zero_point = 0; // Output zero point set at zero by default

                for (0..batch_size) |b| {
                    for (0..num_filters) |f| {
                        var sum: T = 0;
                        const current_group = f / filters_per_group;
                        const input_channel_start = current_group * channels_per_group;
                        const input_channel_end = input_channel_start + channels_per_group;

                        if (kernel_channels_per_group >= 1 and kernel_height >= 1 and kernel_width >= 1) {
                            const center_h = kernel_height / 2;
                            const center_w = kernel_width / 2;

                            // Iterate over relevant input channels for this group
                            for (input_channel_start..input_channel_end) |in_c| {
                                // Map input channel index 'in_c' to the kernel's channel index 'k_c' (0 to C/g - 1)
                                const k_c = in_c - input_channel_start;
                                // Ensure we don't go past the actual input channels
                                if (in_c >= input.shape[1]) continue;
                                // Ensure kernel channel index is valid (should be due to checks, but safety)
                                if (k_c >= kernel_channels_per_group) continue;

                                // Simplified indexing for 1x1 input
                                const input_val_offset = b * input.shape[1] * input.shape[2] * input.shape[3] + in_c * input.shape[2] * input.shape[3]; // Index for input[b, in_c, 0, 0]
                                const input_val = input.data[input_val_offset];

                                // Index for kernel[f, k_c, center_h, center_w]
                                const kernel_val_offset = f * kernel.shape[1] * kernel.shape[2] * kernel.shape[3] + k_c * kernel.shape[2] * kernel.shape[3] + center_h * kernel.shape[3] + center_w;
                                const kernel_val = kernel.data[kernel_val_offset];
                                // sum += input_val * kernel_val;
                                sum += (@as(i32, input_val) - input_zero_point) * (@as(i32, kernel_val) - kernel_zero_point);
                            }
                        }
                        // Add bias if provided
                        const bias_val = if (bias) |bias_tensor| try bias_tensor.get_at(&[_]usize{f}) else 0;

                        // Clamping
                        const accumulated_sum: isize = bias_val + qd.zero_point + ((sum * multiplier) >> (shift + shift_correction));
                        sum = if (accumulated_sum > max_value) {
                            max_value;
                        } else if (accumulated_sum < min_value) {
                            min_value;
                        } else {
                            @as(i32, @intCast(accumulated_sum));
                        };

                        try output.set_at(&[_]usize{ b, f, 0, 0 }, @as(T, @intCast(sum)));
                    }
                }
            },
            else => return TensorError.NotQuantizedTensor,
        }
        return output;
    }
    // --- End Special case ---

    // --- Standard Convolution Calculation ---
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride_h;

    // Calculate output dimensions
    const out_height: usize = @divFloor(in_height - dilated_kernel_h, stride_h) + 1;
    const out_width: usize = @divFloor(in_width - dilated_kernel_w, stride_w) + 1;

    if (out_height <= 0 or out_width <= 0) {
        // Return empty tensor or handle error? For now, error.
        return TensorMathError.InvalidDimensions;
    }

    var output_shape = [_]usize{ batch_size, num_filters, out_height, out_width };
    var output = Tensor(T).fromShape(&pkg_allocator, &output_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("convolve_mem_eff: Failed to allocate output tensor")));
        }
        @panic("Memory allocation failed for output tensor");
    };
    errdefer output.deinit();
    output.details = input.details;

    // Pre-calculate strides for direct pointer access
    const input_h_stride = in_width;
    const input_channel_stride = in_height * in_width;
    const input_batch_stride = in_channels * input_channel_stride;

    const kernel_w_stride = 1;
    const kernel_h_stride = kernel_width;
    const kernel_channel_stride = kernel_height * kernel_width; // Stride for C/g dim
    const kernel_filter_stride = kernel_channels_per_group * kernel_channel_stride; // Stride for M dim

    const output_w_stride = 1;
    const output_h_stride = out_width;
    const output_channel_stride = out_height * out_width; // Stride for M dim
    const output_batch_stride = num_filters * output_channel_stride;

    const Vector = @Vector(4, T);
    // SIMD for floats if kernel width allows and T is float
    const can_use_simd = @typeInfo(T) == .float and kernel_width >= 4;

    // --- Tiling Parameters --- (Can be tuned)
    const TILE_H: usize = 8;
    const TILE_W: usize = 8;
    const UNROLL_FACTOR: usize = 2; // For scalar remainder loop

    // Pointers to data start
    const input_ptr = input.data.ptr;
    const kernel_ptr = kernel.data.ptr;
    const output_ptr = output.data.ptr;
    const bias_ptr = if (bias) |b_tensor| b_tensor.data.ptr else null;

    // Perform convolution using nested loops and pointer arithmetic
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const input_batch_offset = b * input_batch_stride;
        const output_batch_offset = b * output_batch_stride;

        var f: usize = 0;
        while (f < num_filters) : (f += 1) {
            const current_group = f / filters_per_group;
            const kernel_filter_offset = f * kernel_filter_stride; // Offset for filter f [f, _, _, _]
            const output_filter_offset = output_batch_offset + f * output_channel_stride; // Offset for output [b, f, _, _]
            const bias_val: T = if (bias_ptr) |bp| bp[f] else 0;

            // Calculate input channel range for this group
            const input_channel_start = current_group * channels_per_group;
            const input_channel_end = input_channel_start + channels_per_group;

            var oh_tile: usize = 0;
            while (oh_tile < out_height) : (oh_tile += TILE_H) {
                var ow_tile: usize = 0;
                while (ow_tile < out_width) : (ow_tile += TILE_W) {
                    // Calculate bounds for the current tile
                    const oh_end = @min(oh_tile + TILE_H, out_height);
                    const ow_end = @min(ow_tile + TILE_W, out_width);

                    // Innermost loops iterate within the tile
                    var oh = oh_tile;
                    while (oh < oh_end) : (oh += 1) {
                        const h_start = oh * stride_h; // Input h start for this output row

                        var ow = ow_tile;
                        while (ow < ow_end) : (ow += 1) {
                            const w_start = ow * stride_w; // Input w start for this output col
                            var sum_vec: Vector = @splat(0.0);
                            var sum_scalar: T = 0;

                            // Iterate over the input channels relevant to the current group
                            var c = input_channel_start;
                            while (c < input_channel_end) : (c += 1) {
                                const k_c = c - input_channel_start; // Kernel channel index (0 to C/g - 1)

                                const input_base_channel_offset = input_batch_offset + c * input_channel_stride;
                                const kernel_base_channel_offset = kernel_filter_offset + k_c * kernel_channel_stride;

                                var kh: usize = 0;
                                while (kh < kernel_height) : (kh += 1) {
                                    const ih: usize = h_start + kh * dilation_h;
                                    // Skip if input row is out of bounds
                                    if (ih >= in_height) continue;

                                    const input_row_offset = input_base_channel_offset + ih * input_h_stride;
                                    const kernel_row_offset = kernel_base_channel_offset + kh * kernel_h_stride;

                                    // --- SIMD / Scalar Kernel Width Loop ---
                                    if (can_use_simd) {
                                        var kw: usize = 0;
                                        // SIMD part
                                        while (kw + 4 <= kernel_width) : (kw += 4) {
                                            const iw0: usize = w_start + kw * dilation_w;
                                            const iw1: usize = w_start + (kw + 1) * dilation_w;
                                            const iw2: usize = w_start + (kw + 2) * dilation_w;
                                            const iw3: usize = w_start + (kw + 3) * dilation_w;

                                            // Bounds check for input width
                                            if (iw3 < in_width) {
                                                const k_idx = kernel_row_offset + kw * kernel_w_stride; // kw * 1
                                                const k_ptr_offset = kernel_ptr + k_idx;
                                                const k_vec = Vector{ k_ptr_offset[0], k_ptr_offset[1], k_ptr_offset[2], k_ptr_offset[3] };

                                                const i_vec = Vector{
                                                    input_ptr[input_row_offset + iw0],
                                                    input_ptr[input_row_offset + iw1],
                                                    input_ptr[input_row_offset + iw2],
                                                    input_ptr[input_row_offset + iw3],
                                                };
                                                sum_vec = (i_vec * k_vec) + sum_vec;
                                            } else {
                                                // If iw3 is out, break SIMD loop and handle rest with scalar
                                                break;
                                            }
                                        }

                                        // Unrolled scalar part for the remainder after SIMD break or full SIMD loop
                                        var kw_rem = kw; // Start where SIMD left off
                                        while (kw_rem + UNROLL_FACTOR <= kernel_width) : (kw_rem += UNROLL_FACTOR) {
                                            // Unrolled iteration 1
                                            const iw0: usize = w_start + kw_rem * dilation_w;
                                            if (iw0 < in_width) {
                                                const input_val0 = input_ptr[input_row_offset + iw0];
                                                const kernel_val0 = kernel_ptr[kernel_row_offset + kw_rem * kernel_w_stride];
                                                sum_scalar += input_val0 * kernel_val0;
                                            }
                                            // Unrolled iteration 2
                                            const iw1: usize = w_start + (kw_rem + 1) * dilation_w;
                                            if (iw1 < in_width) {
                                                const input_val1 = input_ptr[input_row_offset + iw1];
                                                const kernel_val1 = kernel_ptr[kernel_row_offset + (kw_rem + 1) * kernel_w_stride];
                                                sum_scalar += input_val1 * kernel_val1;
                                            }
                                            // Add more iterations here if UNROLL_FACTOR > 2
                                        }
                                        // Handle any remaining elements after unrolling
                                        while (kw_rem < kernel_width) : (kw_rem += 1) {
                                            const iw: usize = w_start + kw_rem * dilation_w;
                                            if (iw < in_width) {
                                                const input_val = input_ptr[input_row_offset + iw];
                                                const kernel_val = kernel_ptr[kernel_row_offset + kw_rem * kernel_w_stride];
                                                sum_scalar += input_val * kernel_val;
                                            }
                                        }
                                    } else { // Scalar only path
                                        var kw: usize = 0;
                                        while (kw < kernel_width) : (kw += 1) {
                                            const iw: usize = w_start + kw * dilation_w;
                                            if (iw < in_width) { // Check input bounds
                                                const input_val = input_ptr[input_row_offset + iw];
                                                const kernel_val = kernel_ptr[kernel_row_offset + kw * kernel_w_stride];
                                                sum_scalar += input_val * kernel_val;
                                            }
                                        }
                                    }
                                    // --- End SIMD / Scalar Kernel Width Loop ---
                                } // end kh
                            } // end c (channel loop for the group)

                            // Combine SIMD and scalar results
                            var final_sum: T = sum_scalar;
                            if (can_use_simd) {
                                // Horizontal sum of the vector accumulator
                                final_sum += sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
                            }

                            // Add bias and store result
                            const output_idx = output_filter_offset + oh * output_h_stride + ow * output_w_stride;
                            output_ptr[output_idx] = final_sum + bias_val;
                        } // end ow
                    } // end oh
                } // end ow_tile
            } // end oh_tile
        } // end filter loop f
    } // end batch loop b

    return output;
}

pub fn get_convolution_output_shape(input_shape: []const usize, kernel_shape: []const usize, stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, auto_pad: ?[]const u8) ![4]usize {
    std.debug.print("\n =================================================", .{});

    // --- Handle 3D input shape by assuming batch size = 1 ---
    var actual_input_shape: [4]usize = undefined;
    if (input_shape.len == 3) {
        // Assume batch size is 1 if input is 3D (C, H, W)
        actual_input_shape[0] = 1;
        actual_input_shape[1] = input_shape[0]; // Channels
        actual_input_shape[2] = input_shape[1]; // Height
        actual_input_shape[3] = input_shape[2]; // Width
    } else if (input_shape.len == 4) {
        @memcpy(&actual_input_shape, input_shape[0..4]);
    } else {
        std.debug.print("\n[ERROR] get_convolution_output_shape - Invalid input dimensions: {}", .{input_shape.len});
        return TensorMathError.InvalidDimensions;
    }
    // --- End Shape Adjustment ---

    if (kernel_shape.len != 4) {
        std.debug.print("\n[ERROR] get_convolution_output_shape - Invalid kernel dimensions: {}", .{kernel_shape.len});
        return TensorMathError.InvalidDimensions;
    }

    const in_height = actual_input_shape[2];
    const in_width = actual_input_shape[3];
    const out_channels = kernel_shape[0];
    const kernel_height = kernel_shape[2];
    const kernel_width = kernel_shape[3];

    // Set default values for stride and dilation
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride[0];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;

    // Calculate dilated kernel dimensions
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    // Calculate padding and output dimensions
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;
    var expected_out_height: usize = undefined;
    var expected_out_width: usize = undefined;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            expected_out_height = @divFloor(in_height - dilated_kernel_h, stride_h) + 1;
            expected_out_width = @divFloor(in_width - dilated_kernel_w, stride_w) + 1;
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            expected_out_height = ceilDiv(in_height, stride_h);
            expected_out_width = ceilDiv(in_width, stride_w);

            const total_pad_h = @max((expected_out_height - 1) * stride_h + dilated_kernel_h - in_height, 0);
            const total_pad_w = @max((expected_out_width - 1) * stride_w + dilated_kernel_w - in_width, 0);

            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = total_pad_h / 2;
                pad_h_end = total_pad_h - pad_h_begin;
                pad_w_begin = total_pad_w / 2;
                pad_w_end = total_pad_w - pad_w_begin;
            } else {
                pad_h_end = total_pad_h / 2;
                pad_h_begin = total_pad_h - pad_h_end;
                pad_w_end = total_pad_w / 2;
                pad_w_begin = total_pad_w - pad_w_end;
            }
        } else if (std.mem.eql(u8, pad_mode, "NOTSET")) {
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
                } // else default to zero padding
                expected_out_height = @divFloor(in_height + pad_h_begin + pad_h_end - dilated_kernel_h, stride_h) + 1;
                expected_out_width = @divFloor(in_width + pad_w_begin + pad_w_end - dilated_kernel_w, stride_w) + 1;
            } else {
                // Default to VALID padding if pads not provided and NOTSET
                expected_out_height = @divFloor(in_height - dilated_kernel_h, stride_h) + 1;
                expected_out_width = @divFloor(in_width - dilated_kernel_w, stride_w) + 1;
            }
        } else {
            return TensorMathError.InvalidPadding; // Unsupported auto_pad value
        }
    } else {
        // No auto_pad, use explicit padding if provided, otherwise assume zero padding
        if (pads) |p| {
            if (p.len >= 4) {
                pad_h_begin = p[0];
                pad_w_begin = p[1];
                pad_h_end = p[2];
                pad_w_end = p[3];
            } else if (p.len == 2) { // Handle symmetric padding [h, w]
                pad_h_begin = p[0];
                pad_h_end = p[0];
                pad_w_begin = p[1];
                pad_w_end = p[1];
            } // else: pads is non-null but < 2 or 3 length, assume zero padding
        }
        // Calculate output shape with potentially zero padding
        expected_out_height = @divFloor(in_height + pad_h_begin + pad_h_end - dilated_kernel_h, stride_h) + 1;
        expected_out_width = @divFloor(in_width + pad_w_begin + pad_w_end - dilated_kernel_w, stride_w) + 1;
    }

    if (expected_out_height <= 0 or expected_out_width <= 0) {
        std.debug.print("\n[ERROR] get_convolution_output_shape - Calculated output dimensions are non-positive: H={}, W={}", .{ expected_out_height, expected_out_width });
        return TensorMathError.InvalidDimensions;
    }

    // Use the batch size from the (potentially adjusted) input shape
    return [4]usize{ actual_input_shape[0], out_channels, expected_out_height, expected_out_width };
}

// --------------------------------------------------
// --------------------- im2col ---------------------
// --------------------------------------------------
// --------- standard im2col
pub fn im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, dilations: ?[]const usize, group: usize) !Tensor(T) {
    if (input.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }
    // Quantized inputs validation
    if (input.details != .quant)
        return TensorError.NotQuantizedTensor;

    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    if (channels % group != 0) {
        return TensorMathError.InvalidGroupParameter; // Input channels must be divisible by group
    }
    const channels_per_group = channels / group;

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;

    // Calculate effective kernel size considering dilation
    const dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;

    // Check for valid dimensions before calculating output size
    if (height < dilated_kernel_h or width < dilated_kernel_w) {
        if (log_functionC) |log_func| {
            var buf: [256]u8 = undefined;
            _ = std.fmt.bufPrint(&buf, "im2col: Input smaller than dilated kernel. Input HxW: {d}x{d}, Dilated Kernel HxW: {d}x{d}", .{ height, width, dilated_kernel_h, dilated_kernel_w }) catch unreachable;
            log_func(@constCast(@ptrCast(&buf)));
        }
        // Handle this case gracefully, maybe return an empty tensor or a specific error?
        // For now, let's return an error, but calculation below would yield 0 or negative.
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output spatial dimensions
    const out_height = @divFloor(height - dilated_kernel_h, stride_h) + 1;
    const out_width = @divFloor(width - dilated_kernel_w, stride_w) + 1;

    // Calculate the shape of the output im2col matrix
    // Rows: Number of patches = batch_size * out_height * out_width
    // Cols: Size of each flattened patch = channels_per_group * kernel_h * kernel_w * group
    const rows = batch_size * out_height * out_width;
    const cols_per_group = channels_per_group * kernel_h * kernel_w;
    const cols = cols_per_group * group; // Total columns across all groups

    var col_shape = [_]usize{ rows, cols };
    var col_matrix = Tensor(T).fromShape(&pkg_allocator, &col_shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("im2col: Failed to allocate col_matrix tensor")));
        }
        @panic("Memory allocation failed for col_matrix tensor in im2col");
    };
    errdefer col_matrix.deinit();

    // Use the lean version to populate the matrix
    try lean_im2col(T, input, kernel, stride, dilations, group, &col_matrix);
    col_matrix.details = input.details;
    return col_matrix;
}

pub inline fn lean_im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, dilations: ?[]const usize, group: usize, output: *Tensor(T)) !void {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    if (channels % group != 0) {
        return TensorMathError.InvalidGroupParameter; // Input channels must be divisible by group
    }
    const channels_per_group = channels / group;

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;

    const dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;
    const out_height = @divFloor(height - dilated_kernel_h, stride_h) + 1;
    const out_width = @divFloor(width - dilated_kernel_w, stride_w) + 1;

    // Check if output tensor shape matches expected im2col output
    const expected_rows = batch_size * out_height * out_width;
    const expected_cols = channels_per_group * kernel_h * kernel_w;
    if (output.shape.len != 2 or output.shape[0] != expected_rows or output.shape[1] != expected_cols * group) {
        if (log_functionC) |log_func| {
            var buf: [256]u8 = undefined;
            _ = std.fmt.bufPrint(&buf, "lean_im2col: Output shape mismatch. Got {any}, expected [{d}, {d}]", .{ output.shape, expected_rows, expected_cols * group }) catch unreachable;
            log_func(@constCast(@ptrCast(&buf)));
        }
        return TensorMathError.InvalidDimensions; // Output shape mismatch
    }

    // Pre-compute strides
    const input_channel_stride = height * width;
    const input_batch_stride = channels * input_channel_stride;
    const kernel_patch_size = kernel_h * kernel_w; // Size of one kernel patch (h*w)
    const output_col_stride_per_group = channels_per_group * kernel_patch_size; // Stride for one group's columns in output

    const Vector = @Vector(4, T);
    // Check for float type before enabling SIMD
    const can_use_simd = @typeInfo(T) == .float and kernel_w >= 4;

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        const input_batch_offset = b * input_batch_stride;
        const h_start = oh * stride_h;
        const w_start = ow * stride_w;

        var g: usize = 0;
        while (g < group) : (g += 1) {
            const input_group_offset = input_batch_offset + g * channels_per_group * input_channel_stride;
            const output_group_offset = row * output.shape[1] + g * output_col_stride_per_group; // Offset for the start of this group's columns

            var c_in_group: usize = 0;
            while (c_in_group < channels_per_group) : (c_in_group += 1) {
                const input_channel_offset = input_group_offset + c_in_group * input_channel_stride;
                // Output column position calculation: group offset + channel within group offset
                const output_col_base = output_group_offset + c_in_group * kernel_patch_size;

                var kh: usize = 0;
                while (kh < kernel_h) : (kh += 1) {
                    const ih = h_start + kh * dilation_h;
                    // Skip if input h is out of bounds
                    if (ih >= height) {
                        // Fill corresponding output with zeros if input is out of bounds due to dilation/stride
                        const output_row_patch_offset = output_col_base + kh * kernel_w;
                        if (can_use_simd) {
                            var kw_fill: usize = 0;
                            while (kw_fill + 4 <= kernel_w) : (kw_fill += 4) {
                                output.data[output_row_patch_offset + kw_fill] = 0;
                                output.data[output_row_patch_offset + kw_fill + 1] = 0;
                                output.data[output_row_patch_offset + kw_fill + 2] = 0;
                                output.data[output_row_patch_offset + kw_fill + 3] = 0;
                            }
                            while (kw_fill < kernel_w) : (kw_fill += 1) {
                                output.data[output_row_patch_offset + kw_fill] = 0;
                            }
                        } else {
                            for (0..kernel_w) |kw_fill| {
                                output.data[output_row_patch_offset + kw_fill] = 0;
                            }
                        }
                        continue;
                    }

                    const input_row_offset = input_channel_offset + ih * width;
                    const output_row_patch_offset = output_col_base + kh * kernel_w;

                    if (can_use_simd) {
                        var kw: usize = 0;
                        while (kw + 4 <= kernel_w) : (kw += 4) {
                            const iw0 = w_start + kw * dilation_w;
                            const iw1 = w_start + (kw + 1) * dilation_w;
                            const iw2 = w_start + (kw + 2) * dilation_w;
                            const iw3 = w_start + (kw + 3) * dilation_w;
                            const out_idx = output_row_patch_offset + kw;

                            // SIMD bounds check: Ensure all 4 accesses are valid
                            if (iw3 < width) {
                                // TODO not adapted to quantized input
                                const vec = Vector{
                                    input.data[input_row_offset + iw0],
                                    input.data[input_row_offset + iw1],
                                    input.data[input_row_offset + iw2],
                                    input.data[input_row_offset + iw3],
                                };
                                output.data[out_idx] = vec[0];
                                output.data[out_idx + 1] = vec[1];
                                output.data[out_idx + 2] = vec[2];
                                output.data[out_idx + 3] = vec[3];
                            } else {
                                // Handle boundary elements individually if SIMD goes out of bounds
                                output.data[out_idx] = if (iw0 < width) input.data[input_row_offset + iw0] else 0;
                                output.data[out_idx + 1] = if (iw1 < width) input.data[input_row_offset + iw1] else 0;
                                output.data[out_idx + 2] = if (iw2 < width) input.data[input_row_offset + iw2] else 0;
                                output.data[out_idx + 3] = if (iw3 < width) input.data[input_row_offset + iw3] else 0;
                            }
                        }
                        // Handle remaining elements (scalar)
                        var kw_rem = kw;
                        while (kw_rem < kernel_w) : (kw_rem += 1) {
                            const iw = w_start + kw_rem * dilation_w;
                            const out_idx = output_row_patch_offset + kw_rem;
                            output.data[out_idx] = if (iw < width) input.data[input_row_offset + iw] else 0;
                        }
                    } else { // Scalar path
                        var kw: usize = 0;
                        while (kw < kernel_w) : (kw += 1) {
                            const iw = w_start + kw * dilation_w;
                            const out_idx = output_row_patch_offset + kw;
                            // Set output to 0 if input index is out of bounds
                            output.data[out_idx] = if (iw < width) input.data[input_row_offset + iw] else 0;
                        }
                    }
                } // end kh
            } // end c_in_group
        } // end g
    } // end row
}

/// Converts a 2D matrix back to a 4D tensor using col2im algorithm
/// Input shape: [batch_size * out_height * out_width, channels * kernel_height * kernel_width]
/// Output shape: [batch_size, channels, height, width]
pub fn col2im(comptime T: type, col_matrix: *Tensor(T), output_shape: []const usize, kernel: [2]usize, stride: [2]usize) !Tensor(T) {
    if (output_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    var shape: [4]usize = std.mem.zeroes([4]usize);
    @memcpy(&shape, output_shape[0..4]);

    var output = Tensor(T).fromShape(&pkg_allocator, &shape) catch {
        if (log_functionC) |log_func| {
            log_func(@constCast(@ptrCast("col2im: Failed to allocate output tensor")));
        }
        @panic("Memory allocation failed for output tensor in col2im");
    };
    errdefer output.deinit();
    try output.set(0, 0);

    try lean_col2im(T, col_matrix, output_shape, kernel, stride, &output);

    return output;
}

pub inline fn lean_col2im(comptime T: type, col_matrix: *Tensor(T), output_shape: []const usize, kernel: [2]usize, stride: [2]usize, output: *Tensor(T)) !void {
    const batch_size = output_shape[0];
    const channels = output_shape[1];
    const height = output_shape[2];
    const width = output_shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];

    const out_height = @divExact(height - kernel_h, stride_h) + 1;
    const out_width = @divExact(width - kernel_w, stride_w) + 1;

    // Pre-compute strides for faster access
    const output_channel_stride = height * width;
    const output_batch_stride = channels * output_channel_stride;
    const kernel_size = kernel_h * kernel_w;
    const col_channel_stride = kernel_size;
    const col_batch_stride = channels * col_channel_stride;

    // Use SIMD for accumulation when possible
    const Vector = @Vector(4, T);
    const can_use_simd = kernel_w >= 4;

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        // Pre-compute base indices
        const batch_offset = b * output_batch_stride;
        const h_offset = oh * stride_h;
        const w_offset = ow * stride_w;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const channel_offset = batch_offset + c * output_channel_stride;
            const col_offset = row * col_batch_stride + c * col_channel_stride;

            var kh: usize = 0;
            while (kh < kernel_h) : (kh += 1) {
                const h_idx = h_offset + kh;
                if (h_idx >= height) continue;
                const row_offset = channel_offset + h_idx * width;

                if (can_use_simd) {
                    var kw: usize = 0;
                    while (kw + 4 <= kernel_w) : (kw += 4) {
                        const w_idx = w_offset + kw;
                        if (w_idx + 3 >= width) break;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw;

                        // Load current values
                        const curr_vec = Vector{
                            output.data[output_idx],
                            output.data[output_idx + 1],
                            output.data[output_idx + 2],
                            output.data[output_idx + 3],
                        };

                        // Load values to add
                        const add_vec = Vector{
                            col_matrix.data[col_idx],
                            col_matrix.data[col_idx + 1],
                            col_matrix.data[col_idx + 2],
                            col_matrix.data[col_idx + 3],
                        };

                        // Add and store
                        const result = curr_vec + add_vec;
                        output.data[output_idx] = result[0];
                        output.data[output_idx + 1] = result[1];
                        output.data[output_idx + 2] = result[2];
                        output.data[output_idx + 3] = result[3];
                    }

                    // Handle remaining elements
                    var kw_rem = kernel_w - (kernel_w % 4);
                    while (kw_rem < kernel_w) : (kw_rem += 1) {
                        const w_idx = w_offset + kw_rem;
                        if (w_idx >= width) continue;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw_rem;
                        output.data[output_idx] += col_matrix.data[col_idx];
                    }
                } else {
                    var kw: usize = 0;
                    while (kw < kernel_w) : (kw += 1) {
                        const w_idx = w_offset + kw;
                        if (w_idx >= width) continue;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw;
                        output.data[output_idx] += col_matrix.data[col_idx];
                    }
                }
            }
        }
    }
}

// CONVINTEGER --------------------------------------------------------------------------------------------------------------------

/// ONNX ConvInteger operation (Opset 10+)
/// Computes integer convolution: output = sum((x - x_zero_point) * (w - w_zero_point))
/// Accumulation MUST be done in i32. Intermediate product MUST NOT overflow.
/// T1, T2: Input/Weight types (u8 or i8)
/// T3: Output type (must be i32)
/// x: Input tensor [N, C, H, W] (T1)
/// w: Weight tensor [M, C/g, kH, kW] (T2)
/// x_zero_point: Scalar zero point for x (T1)
/// w_zero_point: Scalar or per-output-channel zero point for w (T2)
/// output: Output tensor [N, M, oH, oW] (i32)
pub fn convInteger_lean(
    comptime T1: type, // Input data type (u8 or i8)
    comptime T2: type, // Weight data type (u8 or i8)
    x: *const Tensor(T1),
    w: *const Tensor(T2),
    x_zero_point: ?*const Tensor(T1), // Scalar (shape [])
    w_zero_point: ?*const Tensor(T2), // Scalar (shape []) or 1D (shape [M])
    output: *Tensor(i32), // Output is always i32
    stride: []const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: ?[]const u8,
) !void {
    // --- Basic Type and Dimension Validations ---
    if (@typeInfo(T1) == .int) {
        if (@typeInfo(T1).int.signedness == .unsigned and @typeInfo(T1).int.bits != 8) {
            return error.InvalidDataType; // T1 must be u8 or i8
        }
        if (@typeInfo(T1).int.signedness == .signed and @typeInfo(T1).int.bits != 8) {
            return error.InvalidDataType;
        }
    } else if (@typeInfo(T1) != .ComptimeInt) {
        return error.InvalidDataType; // T1 must be an integer type
    }
    if (@typeInfo(T2) == .int) {
        if (@typeInfo(T2).int.signedness == .unsigned and @typeInfo(T2).int.bits != 8) {
            return error.InvalidDataType; // T2 must be u8 or i8
        }
        if (@typeInfo(T2).int.signedness == .signed and @typeInfo(T2).int.bits != 8) {
            return error.InvalidDataType;
        }
    } else if (@typeInfo(T2) != .ComptimeInt) {
        return error.InvalidDataType; // T2 must be an integer type
    }

    if (x.shape.len != 4 or w.shape.len != 4) {
        return TensorMathError.InvalidDimensions; // Expect 4D input and weight
    }

    // --- Get Dimensions ---
    const batch_size = x.shape[0];
    const in_channels = x.shape[1]; // C
    const in_height = x.shape[2];
    const in_width = x.shape[3];

    const num_filters = w.shape[0]; // M (total output channels)
    const kernel_channels_per_group = w.shape[1]; // C/g
    const kernel_height = w.shape[2]; // kH
    const kernel_width = w.shape[3]; // kW

    // --- Group Validations ---
    const actual_group = group orelse 1;

    if (in_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (num_filters % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (kernel_channels_per_group != in_channels / actual_group) {
        return TensorMathError.InvalidDimensions;
    }
    const channels_per_group = in_channels / actual_group; // C/g
    const filters_per_group = num_filters / actual_group; // M/g

    // --- Zero Point Handling ---
    var x_zp: T1 = 0; // Default zero point is 0
    if (x_zero_point) |zp_tensor| {
        if (zp_tensor.shape.len != 0 and zp_tensor.size != 1) {
            return TensorMathError.InvalidZeroPointShape; // Must be scalar
        }
        x_zp = zp_tensor.data[0];
    }

    var w_zp_scalar: ?T2 = null;
    var w_zp_per_channel: ?[*]const T2 = null;
    if (w_zero_point) |zp_tensor| {
        if (zp_tensor.shape.len == 0 or zp_tensor.size == 1) {
            // Scalar zero point
            w_zp_scalar = zp_tensor.data[0];
        } else if (zp_tensor.shape.len == 1 and zp_tensor.shape[0] == num_filters) {
            // Per-channel zero point
            w_zp_per_channel = zp_tensor.data.ptr;
        } else {
            return TensorMathError.InvalidZeroPointShape; // Must be scalar or 1D with size M
        }
    }

    // --- Stride and Dilation ---
    const stride_h: isize = if (stride.len > 0) @intCast(stride[0]) else 1;
    const stride_w: isize = if (stride.len > 1) @intCast(stride[1]) else stride_h;
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;
    const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

    // --- Padding Calculation ---
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;
    var expected_out_height: usize = undefined;
    var expected_out_width: usize = undefined;

    // Calculate output shape and padding based on auto_pad or explicit pads
    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "VALID")) {
            expected_out_height = @divFloor(in_height - dilated_kernel_h, @as(usize, @intCast(stride_h))) + 1;
            expected_out_width = @divFloor(in_width - dilated_kernel_w, @as(usize, @intCast(stride_w))) + 1;
        } else if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            expected_out_height = ceilDiv(in_height, @as(usize, @intCast(stride_h)));
            expected_out_width = ceilDiv(in_width, @as(usize, @intCast(stride_w)));
            const total_pad_h_calc = @as(isize, @intCast(expected_out_height - 1)) * @as(isize, @intCast(stride_h)) + @as(isize, @intCast(dilated_kernel_h)) - @as(isize, @intCast(in_height));
            const total_pad_w_calc = @as(isize, @intCast(expected_out_width - 1)) * @as(isize, @intCast(stride_w)) + @as(isize, @intCast(dilated_kernel_w)) - @as(isize, @intCast(in_width));
            const total_pad_h = @max(total_pad_h_calc, 0);
            const total_pad_w = @max(total_pad_w_calc, 0);
            if (std.mem.eql(u8, pad_mode, "SAME_UPPER")) {
                pad_h_begin = @as(usize, @intCast(total_pad_h)) / 2;
                pad_h_end = @as(usize, @intCast(total_pad_h)) - pad_h_begin;
                pad_w_begin = @as(usize, @intCast(total_pad_w)) / 2;
                pad_w_end = @as(usize, @intCast(total_pad_w)) - pad_w_begin;
            } else { // SAME_LOWER
                pad_h_end = @as(usize, @intCast(total_pad_h)) / 2;
                pad_h_begin = @as(usize, @intCast(total_pad_h)) - pad_h_end;
                pad_w_end = @as(usize, @intCast(total_pad_w)) / 2;
                pad_w_begin = @as(usize, @intCast(total_pad_w)) - pad_w_end;
            }
        } else if (std.mem.eql(u8, pad_mode, "NOTSET")) {
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
                } // else default zero padding
            } else {} // else default zero padding
            // Calculate output size with determined padding
            const height_calc = @as(isize, @intCast(in_height + pad_h_begin + pad_h_end)) - @as(isize, @intCast(dilated_kernel_h));
            const width_calc = @as(isize, @intCast(in_width + pad_w_begin + pad_w_end)) - @as(isize, @intCast(dilated_kernel_w));
            expected_out_height = @as(usize, @intCast(@divFloor(height_calc, @as(isize, @intCast(stride_h))) + 1));
            expected_out_width = @as(usize, @intCast(@divFloor(width_calc, @as(isize, @intCast(stride_w))) + 1));
        } else {
            return TensorMathError.InvalidPadding; // Unsupported auto_pad value
        }
    } else {
        // No auto_pad, use explicit padding if provided, otherwise assume zero padding
        if (pads) |p| {
            if (p.len >= 4) {
                pad_h_begin = p[0];
                pad_w_begin = p[1];
                pad_h_end = p[2];
                pad_w_end = p[3];
            } else if (p.len == 2) { // Handle symmetric padding [h, w]
                pad_h_begin = p[0];
                pad_h_end = p[0];
                pad_w_begin = p[1];
                pad_w_end = p[1];
            } // else: default zero padding
        }
        // Calculate output shape with potentially zero padding
        const height_calc2 = @as(isize, @intCast(in_height + pad_h_begin + pad_h_end)) - @as(isize, @intCast(dilated_kernel_h));
        const width_calc2 = @as(isize, @intCast(in_width + pad_w_begin + pad_w_end)) - @as(isize, @intCast(dilated_kernel_w));
        expected_out_height = @as(usize, @intCast(@divFloor(height_calc2, @as(isize, @intCast(stride_h))) + 1));
        expected_out_width = @as(usize, @intCast(@divFloor(width_calc2, @as(isize, @intCast(stride_w))) + 1));
    }

    // --- Validate Output Tensor Shape ---

    if (output.shape.len != 4 or
        output.shape[0] != batch_size or
        output.shape[1] != num_filters or
        output.shape[2] != expected_out_height or
        output.shape[3] != expected_out_width)
    {
        return TensorMathError.OutputShapeMismatch;
    }

    // --- Pre-calculate Strides for Direct Pointer Access ---
    const in_h_stride = in_width;
    const in_c_stride = in_height * in_width;
    const in_batch_stride = in_channels * in_c_stride;

    const kernel_w_stride = 1;
    const kernel_h_stride = kernel_width;
    const kernel_c_stride = kernel_height * kernel_width; // Stride for C/g dim
    const kernel_f_stride = kernel_channels_per_group * kernel_c_stride; // Stride for M dim

    const out_w_stride = 1;
    const out_h_stride = expected_out_width;
    const out_c_stride = expected_out_height * out_h_stride; // Stride for M dim
    const out_batch_stride = num_filters * out_c_stride;

    // --- Pointers to Data Start ---
    const x_ptr = x.data.ptr;
    const w_ptr = w.data.ptr;
    const y_ptr = output.data.ptr; // Output data pointer

    // --- Convolution Loop ---
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const in_batch_offset = b * in_batch_stride;
        const out_batch_offset = b * out_batch_stride;

        var f: usize = 0;
        while (f < num_filters) : (f += 1) {
            const current_group = f / filters_per_group;
            const w_f_offset = f * kernel_f_stride; // Offset for filter f in w [f, _, _, _]
            const y_f_offset = out_batch_offset + f * out_c_stride; // Offset for output [b, f, _, _]

            // Get the correct weight zero point for this filter
            const w_zp: T2 = if (w_zp_per_channel) |zp_ptr| zp_ptr[f] else (w_zp_scalar orelse 0);

            // Calculate input channel range for this group
            const in_c_start = current_group * channels_per_group;
            const in_c_end = in_c_start + channels_per_group;

            var oh: isize = 0;
            while (oh < expected_out_height) : (oh += 1) {
                const ih_base = oh * @as(isize, @intCast(stride_h)) - @as(isize, @intCast(pad_h_begin)); // Input h start potentially negative due to padding

                var ow: usize = 0;
                while (ow < expected_out_width) : (ow += 1) {
                    const iw_base = @as(isize, @intCast(ow)) * @as(isize, @intCast(stride_w)) - @as(isize, @intCast(pad_w_begin)); // Input w start potentially negative
                    var accumulator: i32 = 0; // Accumulate in i32

                    // Iterate over the input channels relevant to the current group
                    var c = in_c_start;
                    while (c < in_c_end) : (c += 1) {
                        const k_c = c - in_c_start; // Kernel channel index (0 to C/g - 1)
                        const in_c_offset = in_batch_offset + c * in_c_stride;
                        const w_c_offset = w_f_offset + k_c * kernel_c_stride;

                        var kh: usize = 0;
                        while (kh < kernel_height) : (kh += 1) {
                            const ih: isize = ih_base + @as(isize, @intCast(kh * dilation_h));

                            var kw: usize = 0;
                            while (kw < kernel_width) : (kw += 1) {
                                const iw: isize = iw_base + @as(isize, @intCast(kw * dilation_w));

                                // Check if the input coordinates (ih, iw) are within the valid (unpadded) input bounds
                                if (ih >= 0 and ih < @as(isize, @intCast(in_height)) and iw >= 0 and iw < @as(isize, @intCast(in_width))) {
                                    // Valid input pixel
                                    const x_idx = in_c_offset + @as(usize, @intCast(ih)) * in_h_stride + @as(usize, @intCast(iw));
                                    const w_idx = w_c_offset + kh * kernel_h_stride + kw * kernel_w_stride;

                                    const x_val: T1 = x_ptr[x_idx];
                                    const w_val: T2 = w_ptr[w_idx];

                                    // Perform calculation: (x - x_zp) * (w - w_zp)
                                    // Cast to i32 BEFORE operations to prevent overflow during multiplication
                                    const term: i32 = (@as(i32, x_val) - @as(i32, x_zp)) * (@as(i32, w_val) - @as(i32, w_zp));
                                    accumulator += term;
                                } else {
                                    // Padded area: contribution is (-x_zp) * (w - w_zp)
                                    const w_idx = w_c_offset + kh * kernel_h_stride + kw * kernel_w_stride;
                                    const w_val: T2 = w_ptr[w_idx];
                                    // Cast to i32 before operations
                                    const term: i32 = (@as(i32, 0) - @as(i32, x_zp)) * (@as(i32, w_val) - @as(i32, w_zp));
                                    accumulator += term;
                                }
                            } // kw
                        } // kh
                    } // c (channel loop for the group)

                    // Store the final accumulated i32 result
                    const y_idx = y_f_offset + @as(usize, @intCast(oh)) * out_h_stride + ow * out_w_stride;
                    y_ptr[y_idx] = accumulator;
                } // ow
            } // oh
        } // f (filter loop)
    } // b (batch loop)

}
