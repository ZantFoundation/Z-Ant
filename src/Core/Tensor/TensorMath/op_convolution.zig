const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const op_mat_mul = @import("op_mat_mul.zig");
const mat_mul = op_mat_mul.mat_mul;
const op_padding = @import("lib_shape_math/op_padding.zig");
const addPaddingAndDilation = op_padding.addPaddingAndDilation;

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
///
pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void {
    // Input validation
    if (input.shape.len != 4 or kernel.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = kernel.shape[0];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    // Group validation - currently only supporting group=1
    if (group != null and group.? != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Set default values for stride and dilation
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride[0];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;

    // Calculate padding
    var pad_h_begin: usize = 0;
    var pad_h_end: usize = 0;
    var pad_w_begin: usize = 0;
    var pad_w_end: usize = 0;
    var expected_out_height: usize = in_height;
    var expected_out_width: usize = in_width;

    std.debug.print("\n[DEBUG] OnnxConvLean - Initial input shape: {any}", .{input.shape});
    std.debug.print("\n[DEBUG] OnnxConvLean - Kernel shape: {any}", .{kernel.shape});
    std.debug.print("\n[DEBUG] OnnxConvLean - Stride: {any}", .{stride});
    std.debug.print("\n[DEBUG] OnnxConvLean - Dilations: {any}", .{dilations});

    if (auto_pad) |pad_mode| {
        const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
        const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;

        std.debug.print("\n[DEBUG] OnnxConvLean - Auto pad mode: {s}", .{pad_mode});
        std.debug.print("\n[DEBUG] OnnxConvLean - Dilated kernel: h={}, w={}", .{ dilated_kernel_h, dilated_kernel_w });

        if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            const out_height = in_height;
            const out_width = in_width;
            expected_out_height = out_height;
            expected_out_width = out_width;

            // Calculate total padding needed to maintain input dimensions
            const total_pad_h = (dilated_kernel_h - 1) * stride_h;
            const total_pad_w = (dilated_kernel_w - 1) * stride_w;

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
        }
    } else if (pads) |p| {
        if (p.len >= 4) {
            pad_h_begin = p[0];
            pad_w_begin = p[1];
            pad_h_end = p[2];
            pad_w_end = p[3];
        }
    }

    // Create a copy of input for padding
    var padded_input = try input.copy();
    defer padded_input.deinit();

    std.debug.print("\n[DEBUG] OnnxConvLean - Final padding values:", .{});
    std.debug.print("\n  h_begin={}, h_end={}", .{ pad_h_begin, pad_h_end });
    std.debug.print("\n  w_begin={}, w_end={}", .{ pad_w_begin, pad_w_end });

    // Initialize padded input with zeros
    var padded_shape = [_]usize{ input.shape[0], input.shape[1], input.shape[2] + pad_h_begin + pad_h_end, input.shape[3] + pad_w_begin + pad_w_end };

    // Create new padded tensor
    padded_input.deinit(); // Clean up the copy we don't need
    padded_input = try Tensor(T).fromShape(&pkg_allocator, &padded_shape);
    try padded_input.set(0, 0); // Initialize to zeros

    // Copy original input to center of padded tensor
    for (0..input.shape[0]) |b| {
        for (0..input.shape[1]) |c| {
            for (0..input.shape[2]) |h| {
                for (0..input.shape[3]) |w| {
                    const val = try input.get_at(&[_]usize{ b, c, h, w });
                    try padded_input.set_at(&[_]usize{ b, c, h + pad_h_begin, w + pad_w_begin }, val);
                }
            }
        }
    }

    std.debug.print("\n[DEBUG] OnnxConvLean - Padded input shape: {any}", .{padded_input.shape});

    // If bias is not provided, create a zero bias tensor
    var zero_bias: Tensor(T) = undefined;
    if (bias == null) {
        var bias_shape = [_]usize{out_channels};
        zero_bias = try Tensor(T).fromShape(&pkg_allocator, &bias_shape);
        errdefer zero_bias.deinit();
        try zero_bias.set(0, 0);
    }

    // Create stride and dilation arrays
    var stride_arr = [_]usize{ stride_h, stride_w };
    var dilation_arr = [_]usize{ dilation_h, dilation_w };

    // Call convolve_tensor_with_bias with the dilation parameter
    var result = try convolve_tensor_with_bias(
        T,
        &padded_input,
        kernel,
        if (bias) |b| b else &zero_bias,
        &stride_arr,
        &dilation_arr,
    );
    defer result.deinit();

    if (bias == null) {
        zero_bias.deinit();
    }

    // For SAME_UPPER padding, ensure output shape matches input shape
    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            if (result.shape[2] != expected_out_height or result.shape[3] != expected_out_width) {
                var shape = [_]usize{ input.shape[0], out_channels, expected_out_height, expected_out_width };

                // Validate output tensor shape
                if (!std.mem.eql(usize, &shape, output.shape[0..4])) {
                    return TensorMathError.InvalidDimensions;
                }

                // Calculate center offsets
                const h_offset = @divFloor(result.shape[2] - expected_out_height, 2);
                const w_offset = @divFloor(result.shape[3] - expected_out_width, 2);

                // Copy the center portion of the result
                for (0..input.shape[0]) |b| {
                    for (0..out_channels) |c| {
                        for (0..expected_out_height) |h| {
                            for (0..expected_out_width) |w| {
                                const src_h = h + h_offset;
                                const src_w = w + w_offset;
                                if (src_h < result.shape[2] and src_w < result.shape[3]) {
                                    const val = try result.get_at(&[_]usize{ b, c, src_h, src_w });
                                    try output.set_at(&[_]usize{ b, c, h, w }, val);
                                } else {
                                    try output.set_at(&[_]usize{ b, c, h, w }, 0);
                                }
                            }
                        }
                    }
                }
                return;
            }
        }
    }

    // Copy result to output tensor
    if (!std.mem.eql(usize, result.shape[0..4], output.shape[0..4])) {
        return TensorMathError.InvalidDimensions;
    }
    @memcpy(output.data, result.data);
}

pub fn convolve_tensor_with_bias(
    comptime T: type,
    input: *Tensor(T),
    kernel: *const Tensor(T),
    bias: ?*const Tensor(T),
    stride: []const usize, // shape:[row_stride, column_stride]
    dilations: ?[]const usize, // shape:[row_dilation, column_dilation]
) !Tensor(T) {
    const nDimInput = input.shape.len;
    const nDimKernel = kernel.shape.len;

    std.debug.print("\n[DEBUG] convolve_tensor_with_bias - Starting", .{});
    std.debug.print("\n[DEBUG] Input dimensions: {}", .{nDimInput});
    std.debug.print("\n[DEBUG] Kernel dimensions: {}", .{nDimKernel});

    if (nDimInput != 4 or nDimKernel != 4) {
        std.debug.print("\n[DEBUG] Invalid dimensions - nDimInput: {}, nDimKernel: {}", .{ nDimInput, nDimKernel });
        return TensorMathError.InvalidDimensions;
    }

    if (nDimKernel > nDimInput) {
        std.debug.print("\n[DEBUG] Kernel dimensions larger than input - nDimKernel: {}, nDimInput: {}", .{ nDimKernel, nDimInput });
        return TensorMathError.InputTensorDifferentShape;
    }

    std.debug.print("\n[DEBUG] Input channels: {}, Kernel channels: {}", .{ input.shape[nDimInput - 3], kernel.shape[nDimKernel - 3] });
    if (input.shape[nDimInput - 3] != kernel.shape[nDimKernel - 3]) {
        std.debug.print("\n[DEBUG] Channel mismatch - input: {}, kernel: {}", .{ input.shape[nDimInput - 3], kernel.shape[nDimKernel - 3] });
        return TensorMathError.InputTensorsWrongShape;
    }

    if (bias != null) {
        const bias_dim = bias.?.shape.len;
        std.debug.print("\n[DEBUG] Bias dimensions: {}, Kernel filters: {}", .{ bias_dim, kernel.shape[nDimKernel - 4] });
        if (bias_dim < 1 or bias.?.shape[bias_dim - 1] != kernel.shape[nDimKernel - 4]) {
            std.debug.print("\n[DEBUG] Invalid bias dimensions - bias_dim: {}, expected: {}", .{ bias_dim, kernel.shape[nDimKernel - 4] });
            return TensorMathError.InvalidDimensions;
        }
    }

    if (stride.len != 2 or stride[0] == 0 or stride[1] == 0) {
        std.debug.print("\n[DEBUG] Invalid stride - len: {}, values: {any}", .{ stride.len, stride });
        return TensorMathError.WrongStride;
    }

    std.debug.print("\n[DEBUG] Input shape: {any}", .{input.shape});
    std.debug.print("\n[DEBUG] Kernel shape: {any}", .{kernel.shape});
    std.debug.print("\n[DEBUG] Dilations: {any}", .{dilations});
    std.debug.print("\n[DEBUG] Stride: {any}", .{stride});

    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;
    const dilated_kernel_h = (kernel.shape[nDimKernel - 2] - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel.shape[nDimKernel - 1] - 1) * dilation_w + 1;

    std.debug.print("\n[DEBUG] Dilated kernel dimensions: h={}, w={}", .{ dilated_kernel_h, dilated_kernel_w });
    std.debug.print("\n[DEBUG] Input dimensions: h={}, w={}", .{ input.shape[nDimInput - 2], input.shape[nDimInput - 1] });

    // Calculate output dimensions
    const out_height = if (input.shape[nDimInput - 2] >= dilated_kernel_h)
        @divFloor(input.shape[nDimInput - 2] - dilated_kernel_h + stride[0], stride[0])
    else
        0;
    const out_width = if (input.shape[nDimInput - 1] >= dilated_kernel_w)
        @divFloor(input.shape[nDimInput - 1] - dilated_kernel_w + stride[1], stride[1])
    else
        0;

    std.debug.print("\n[DEBUG] Calculated output dimensions: h={}, w={}", .{ out_height, out_width });

    if (out_height == 0 or out_width == 0) {
        std.debug.print("\n[DEBUG] Invalid output dimensions - height: {}, width: {}", .{ out_height, out_width });
        return TensorMathError.InvalidDimensions;
    }

    std.debug.print("\n[DEBUG] Creating output tensor with shape: [{}, {}, {}, {}]", .{ input.shape[0], kernel.shape[0], out_height, out_width });

    var output_shape = [_]usize{ input.shape[0], kernel.shape[0], out_height, out_width };
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    const kernel_size = [2]usize{ kernel.shape[nDimKernel - 2], kernel.shape[nDimKernel - 1] };
    const stride_size = [2]usize{ stride[0], stride[1] };

    std.debug.print("\n[DEBUG] Calling im2col", .{});
    var input_col = try im2col(T, input, kernel_size, stride_size, dilations);
    defer input_col.deinit();
    std.debug.print("\n[DEBUG] im2col completed. Shape: {any}", .{input_col.shape});

    const num_filters = kernel.shape[0];
    const kernel_elements = try std.math.mul(usize, kernel.shape[1], try std.math.mul(usize, kernel.shape[2], kernel.shape[3]));

    var kernel_matrix_shape = [_]usize{ kernel_elements, num_filters };
    var kernel_matrix = try Tensor(T).fromShape(&pkg_allocator, &kernel_matrix_shape);
    defer kernel_matrix.deinit();

    std.debug.print("\n[DEBUG] Reshaping kernel. Shape: {any}", .{kernel_matrix_shape});

    // Safely copy kernel data
    for (0..num_filters) |f| {
        for (0..kernel_elements) |i| {
            const idx = try std.math.mul(usize, f, kernel_elements);
            try kernel_matrix.set_at(&[_]usize{ i, f }, kernel.data[idx + i]);
        }
    }

    std.debug.print("\n[DEBUG] Performing matrix multiplication", .{});
    var result = try mat_mul(T, &input_col, &kernel_matrix);
    defer result.deinit();
    std.debug.print("\n[DEBUG] Matrix multiplication completed. Shape: {any}", .{result.shape});

    // Safe copy with direct floating point addition
    var idx: usize = 0;
    for (0..input.shape[0]) |b| {
        for (0..out_height) |h| {
            for (0..out_width) |w| {
                for (0..num_filters) |f| {
                    const val = try result.get_at(&[_]usize{ idx, f });
                    const bias_val = if (bias) |bias_tensor| bias_tensor.data[f] else 0;
                    try output.set_at(&[_]usize{ b, f, h, w }, val + bias_val);
                }
                idx += 1;
            }
        }
    }

    std.debug.print("\n[DEBUG] Convolution completed successfully", .{});
    return output;
}

pub fn get_convolution_output_shape(input_shape: []const usize, kernel_shape: []const usize, stride: []const usize) ![4]usize {
    if (input_shape.len != 4 or kernel_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    if (stride.len != 2 or stride[0] == 0 or stride[1] == 0) {
        return TensorMathError.WrongStride;
    }

    const batch_size = input_shape[0];
    const num_filters = kernel_shape[0];
    const out_height = try std.math.divExact(usize, input_shape[2] - kernel_shape[2], stride[0]) + 1;
    const out_width = try std.math.divExact(usize, input_shape[3] - kernel_shape[3], stride[1]) + 1;

    return [4]usize{ batch_size, num_filters, out_height, out_width };
}

pub fn convolution_backward_biases(comptime T: type, dValues: *Tensor(T)) !Tensor(T) {
    if (dValues.shape.len != 4) return TensorMathError.InvalidDimensions;

    const out_channels = dValues.shape[1];
    var bias_gradients_shape = [_]usize{out_channels};

    var bias_gradients = try Tensor(T).fromShape(&pkg_allocator, &bias_gradients_shape);
    errdefer bias_gradients.deinit();
    try bias_gradients.set(0, 0);

    const batch_size = dValues.shape[0];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    for (0..out_channels) |oc| {
        var sum: T = 0;
        for (0..batch_size) |b| {
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const val = try dValues.get_at(&[_]usize{ b, oc, h, w });
                    sum += val; // Direct addition for floating point
                }
            }
        }
        try bias_gradients.set_at(&[_]usize{oc}, sum);
    }

    return bias_gradients;
}

pub fn convolution_backward_weights(comptime T: type, input: *Tensor(T), dvalues: *Tensor(T), kernel_shape: []const usize, stride: [2]usize) !Tensor(T) {
    if (kernel_shape.len != 4) return TensorMathError.InvalidDimensions;

    // Create mutable copy of kernel_shape
    var mutable_kernel_shape: [4]usize = undefined;
    @memcpy(&mutable_kernel_shape, kernel_shape);

    const batch_size = input.shape[0];
    const num_filters = kernel_shape[0];
    const kernel_height = kernel_shape[2];
    const kernel_width = kernel_shape[3];

    const kernel_size = [2]usize{ kernel_height, kernel_width };
    var input_col = try im2col(T, input, kernel_size, stride, null);
    defer input_col.deinit();

    const out_height = dvalues.shape[2];
    const out_width = dvalues.shape[3];
    const total_spatial = out_height * out_width;

    var dval_shape = [_]usize{ num_filters, batch_size * total_spatial };
    var dval_reshaped = try Tensor(T).fromShape(&pkg_allocator, &dval_shape);
    defer dval_reshaped.deinit();

    // Pre-compute strides for faster access
    const out_channel_stride = out_height * out_width;
    const out_batch_stride = num_filters * out_channel_stride;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = total_spatial >= 4;

    // Safe copy with optimized memory access
    for (0..batch_size) |b| {
        const batch_offset = b * out_batch_stride;
        const dst_batch_offset = b * total_spatial;

        for (0..num_filters) |f| {
            const src_offset = batch_offset + f * out_channel_stride;
            const dst_offset = f * batch_size * total_spatial + dst_batch_offset;

            if (can_use_simd) {
                var i: usize = 0;
                while (i + 4 <= total_spatial) : (i += 4) {
                    const vec = Vector{
                        dvalues.data[src_offset + i],
                        dvalues.data[src_offset + i + 1],
                        dvalues.data[src_offset + i + 2],
                        dvalues.data[src_offset + i + 3],
                    };

                    dval_reshaped.data[dst_offset + i] = vec[0];
                    dval_reshaped.data[dst_offset + i + 1] = vec[1];
                    dval_reshaped.data[dst_offset + i + 2] = vec[2];
                    dval_reshaped.data[dst_offset + i + 3] = vec[3];
                }

                // Handle remaining elements
                var i_rem = total_spatial - (total_spatial % 4);
                while (i_rem < total_spatial) : (i_rem += 1) {
                    dval_reshaped.data[dst_offset + i_rem] = dvalues.data[src_offset + i_rem];
                }
            } else {
                for (0..total_spatial) |i| {
                    dval_reshaped.data[dst_offset + i] = dvalues.data[src_offset + i];
                }
            }
        }
    }

    var dW = try mat_mul(T, &dval_reshaped, &input_col);
    defer dW.deinit();

    // Use SIMD for division
    const batch_size_f = @as(T, @floatFromInt(batch_size));
    if (can_use_simd) {
        const batch_size_vec = Vector{ batch_size_f, batch_size_f, batch_size_f, batch_size_f };
        var i: usize = 0;
        while (i + 4 <= dW.data.len) : (i += 4) {
            const vec = Vector{
                dW.data[i],
                dW.data[i + 1],
                dW.data[i + 2],
                dW.data[i + 3],
            };
            const result = vec / batch_size_vec;
            dW.data[i] = result[0];
            dW.data[i + 1] = result[1];
            dW.data[i + 2] = result[2];
            dW.data[i + 3] = result[3];
        }

        // Handle remaining elements
        var i_rem = dW.data.len - (dW.data.len % 4);
        while (i_rem < dW.data.len) : (i_rem += 1) {
            dW.data[i_rem] = dW.data[i_rem] / batch_size_f;
        }
    } else {
        for (dW.data) |*val| {
            val.* = val.* / batch_size_f;
        }
    }

    var dW_reshaped = try Tensor(T).fromShape(&pkg_allocator, &mutable_kernel_shape);
    errdefer dW_reshaped.deinit();

    @memcpy(dW_reshaped.data, dW.data);

    return dW_reshaped;
}

pub fn convolution_backward_input(comptime T: type, dvalues: *const Tensor(T), kernel: *const Tensor(T), input_shape: []const usize, stride: [2]usize) !Tensor(T) {
    if (input_shape.len != 4) return TensorMathError.InvalidDimensions;

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];
    const num_filters = kernel.shape[0];

    const out_height = dvalues.shape[2];
    const out_width = dvalues.shape[3];
    const total_spatial = out_height * out_width;

    const total_batch_spatial = try std.math.mul(usize, batch_size, total_spatial);
    var dval_shape = [_]usize{ total_batch_spatial, num_filters };
    var dval_reshaped = try Tensor(T).fromShape(&pkg_allocator, &dval_shape);
    defer dval_reshaped.deinit();

    // Pre-compute strides for faster access
    const dval_channel_stride = out_height * out_width;
    const dval_batch_stride = num_filters * dval_channel_stride;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = num_filters >= 4;

    // Safe reshape with optimized memory access
    for (0..batch_size) |b| {
        const batch_offset = b * dval_batch_stride;
        const dst_batch_offset = b * total_spatial;

        for (0..total_spatial) |i| {
            const src_base = batch_offset + i % dval_channel_stride;
            const dst_base = dst_batch_offset + i;

            if (can_use_simd) {
                var f: usize = 0;
                while (f + 4 <= num_filters) : (f += 4) {
                    const vec = Vector{
                        dvalues.data[src_base + f * dval_channel_stride],
                        dvalues.data[src_base + (f + 1) * dval_channel_stride],
                        dvalues.data[src_base + (f + 2) * dval_channel_stride],
                        dvalues.data[src_base + (f + 3) * dval_channel_stride],
                    };

                    dval_reshaped.data[dst_base * num_filters + f] = vec[0];
                    dval_reshaped.data[dst_base * num_filters + f + 1] = vec[1];
                    dval_reshaped.data[dst_base * num_filters + f + 2] = vec[2];
                    dval_reshaped.data[dst_base * num_filters + f + 3] = vec[3];
                }

                // Handle remaining filters
                var f_rem = num_filters - (num_filters % 4);
                while (f_rem < num_filters) : (f_rem += 1) {
                    dval_reshaped.data[dst_base * num_filters + f_rem] = dvalues.data[src_base + f_rem * dval_channel_stride];
                }
            } else {
                for (0..num_filters) |f| {
                    dval_reshaped.data[dst_base * num_filters + f] = dvalues.data[src_base + f * dval_channel_stride];
                }
            }
        }
    }

    const kernel_spatial = kernel_height * kernel_width;
    var transposed_shape = [_]usize{ num_filters, try std.math.mul(usize, channels, kernel_spatial) };
    var kernel_transposed = try Tensor(T).fromShape(&pkg_allocator, &transposed_shape);
    defer kernel_transposed.deinit();

    // Pre-compute strides for kernel transpose
    const kernel_filter_stride = channels * kernel_spatial;

    // Safe transpose with optimized memory access
    for (0..num_filters) |f| {
        const filter_offset = f * kernel_filter_stride;
        const src_filter_offset = f * channels * kernel_spatial;

        for (0..channels) |c| {
            const channel_offset = filter_offset + c * kernel_spatial;
            const src_channel_offset = src_filter_offset + c * kernel_spatial;

            if (can_use_simd and kernel_spatial >= 4) {
                var k: usize = 0;
                while (k + 4 <= kernel_spatial) : (k += 4) {
                    const vec = Vector{
                        kernel.data[src_channel_offset + k],
                        kernel.data[src_channel_offset + k + 1],
                        kernel.data[src_channel_offset + k + 2],
                        kernel.data[src_channel_offset + k + 3],
                    };

                    kernel_transposed.data[channel_offset + k] = vec[0];
                    kernel_transposed.data[channel_offset + k + 1] = vec[1];
                    kernel_transposed.data[channel_offset + k + 2] = vec[2];
                    kernel_transposed.data[channel_offset + k + 3] = vec[3];
                }

                // Handle remaining elements
                var k_rem = kernel_spatial - (kernel_spatial % 4);
                while (k_rem < kernel_spatial) : (k_rem += 1) {
                    kernel_transposed.data[channel_offset + k_rem] = kernel.data[src_channel_offset + k_rem];
                }
            } else {
                for (0..kernel_spatial) |k| {
                    kernel_transposed.data[channel_offset + k] = kernel.data[src_channel_offset + k];
                }
            }
        }
    }

    var dX_col = try mat_mul(T, &dval_reshaped, &kernel_transposed);
    defer dX_col.deinit();

    const kernel_size = [2]usize{ kernel_height, kernel_width };
    return try col2im(T, &dX_col, input_shape, kernel_size, stride);
}

// --------------------------------------------------
// --------------------- im2col ---------------------
// --------------------------------------------------
// --------- standard im2col
pub fn im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, dilations: ?[]const usize) !Tensor(T) {
    if (input.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }

    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;

    if (height < kernel_h or width < kernel_w) {
        return TensorMathError.InvalidDimensions;
    }

    const dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;
    const out_height = (height - dilated_kernel_h + 1) / stride_h;
    const out_width = (width - dilated_kernel_w + 1) / stride_w;

    const rows = try std.math.mul(usize, batch_size, try std.math.mul(usize, out_height, out_width));
    const cols = try std.math.mul(usize, channels, try std.math.mul(usize, kernel_h, kernel_w));

    var col_shape = [_]usize{ rows, cols };
    var col_matrix = try Tensor(T).fromShape(&pkg_allocator, &col_shape);
    errdefer col_matrix.deinit();

    try lean_im2col(T, input, kernel, stride, dilations, &col_matrix);

    return col_matrix;
}
// --------- lean im2col
pub inline fn lean_im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, dilations: ?[]const usize, output: *Tensor(T)) !void {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else 1 else 1;

    const dilated_kernel_h = (kernel_h - 1) * dilation_h + 1;
    const dilated_kernel_w = (kernel_w - 1) * dilation_w + 1;
    const out_height = (height - dilated_kernel_h + 1) / stride_h;
    const out_width = (width - dilated_kernel_w + 1) / stride_w;

    // Pre-compute strides for faster access
    const input_channel_stride = height * width;
    const input_batch_stride = channels * input_channel_stride;
    const kernel_size = kernel_h * kernel_w;
    const output_col_stride = channels * kernel_size;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = kernel_w >= 4;

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        // Pre-compute base indices
        const batch_offset = b * input_batch_stride;
        const h_offset = oh * stride_h;
        const w_offset = ow * stride_w;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const channel_offset = batch_offset + c * input_channel_stride;
            const output_offset = row * output_col_stride + c * kernel_size;

            var kh: usize = 0;
            while (kh < kernel_h) : (kh += 1) {
                const h_idx = h_offset + kh * dilation_h;
                const row_offset = channel_offset + h_idx * width;

                if (can_use_simd) {
                    var kw: usize = 0;
                    while (kw + 4 <= kernel_w) : (kw += 4) {
                        const w_idx = w_offset + kw * dilation_w;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw;

                        const vec = Vector{
                            input.data[input_idx],
                            input.data[input_idx + dilation_w],
                            input.data[input_idx + 2 * dilation_w],
                            input.data[input_idx + 3 * dilation_w],
                        };

                        output.data[out_idx] = vec[0];
                        output.data[out_idx + 1] = vec[1];
                        output.data[out_idx + 2] = vec[2];
                        output.data[out_idx + 3] = vec[3];
                    }

                    // Handle remaining elements
                    var kw_rem = kernel_w - (kernel_w % 4);
                    while (kw_rem < kernel_w) : (kw_rem += 1) {
                        const w_idx = w_offset + kw_rem * dilation_w;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw_rem;
                        output.data[out_idx] = input.data[input_idx];
                    }
                } else {
                    var kw: usize = 0;
                    while (kw < kernel_w) : (kw += 1) {
                        const w_idx = w_offset + kw * dilation_w;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw;
                        output.data[out_idx] = input.data[input_idx];
                    }
                }
            }
        }
    }
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

    var output = try Tensor(T).fromShape(&pkg_allocator, &shape);
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

    const out_height = try std.math.divExact(usize, height - kernel_h, stride_h) + 1;
    const out_width = try std.math.divExact(usize, width - kernel_w, stride_w) + 1;

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

pub fn OnnxConv(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !Tensor(T) {
    // Input validation
    if (input.shape.len != 4 or kernel.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = kernel.shape[0];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    // Group validation - currently only supporting group=1
    if (group != null and group.? != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Set default values for stride and dilation
    const stride_h = if (stride.len > 0) stride[0] else 1;
    const stride_w = if (stride.len > 1) stride[1] else stride[0];
    const dilation_h = if (dilations) |d| if (d.len > 0) d[0] else 1 else 1;
    const dilation_w = if (dilations) |d| if (d.len > 1) d[1] else d[0] else 1;

    // Calculate output dimensions
    var expected_out_height: usize = in_height;
    var expected_out_width: usize = in_width;

    if (auto_pad) |pad_mode| {
        if (std.mem.eql(u8, pad_mode, "SAME_UPPER") or std.mem.eql(u8, pad_mode, "SAME_LOWER")) {
            expected_out_height = in_height;
            expected_out_width = in_width;
        }
    } else {
        const dilated_kernel_h = (kernel_height - 1) * dilation_h + 1;
        const dilated_kernel_w = (kernel_width - 1) * dilation_w + 1;
        const total_pad_h = if (pads) |p| if (p.len >= 4) p[0] + p[2] else 0 else 0;
        const total_pad_w = if (pads) |p| if (p.len >= 4) p[1] + p[3] else 0 else 0;

        expected_out_height = (in_height + total_pad_h - dilated_kernel_h) / stride_h + 1;
        expected_out_width = (in_width + total_pad_w - dilated_kernel_w) / stride_w + 1;
    }

    // Create output tensor with correct shape
    var output_shape = [_]usize{ input.shape[0], out_channels, expected_out_height, expected_out_width };
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Call the lean version
    try OnnxConvLean(T, input, kernel, &output, bias, stride, pads, dilations, group, auto_pad);

    return output;
}
