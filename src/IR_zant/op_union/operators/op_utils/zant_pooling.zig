const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>> DEPRECATED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// use op_maxPool, op_averagePool, op_globalAveragePool...

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

pub const AutoPadType = enum {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

pub fn get_pooling_output_shape(input_shape: []const usize, kernel: [2]usize, stride: [2]usize) ![4]usize {
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Input shape: {any}", .{input_shape});
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Kernel: {any}", .{kernel});
    // std.log.debug("\n[DEBUG] get_pooling_output_shape - Stride: {any}", .{stride});

    if (input_shape.len != 4) {
        std.log.warn("\n[DEBUG] ERROR: Invalid dimensions - input shape length is not 4", .{});
        return TensorMathError.InvalidDimensions;
    }

    if (kernel[0] == 0 or kernel[1] == 0 or stride[0] == 0 or stride[1] == 0) {
        std.log.warn("\n[DEBUG] ERROR: Wrong stride - kernel or stride contains zeros", .{});
        return TensorMathError.WrongStride;
    }

    if (kernel[0] > input_shape[2] or kernel[1] > input_shape[3]) {
        std.log.warn("\n[DEBUG] WARNING: Kernel size ({},{}) > input dimensions ({},{})", .{ kernel[0], kernel[1], input_shape[2], input_shape[3] });

        // SPECIAL CASE: Instead of error, return a shape with 1x1 spatial dimensions
        // when kernel is larger than input
        std.log.debug("\n[DEBUG] Using special case: returning output with 1x1 spatial dimensions", .{});
        const batch_size = input_shape[0];
        const channels = input_shape[1];

        return [4]usize{ batch_size, channels, 1, 1 };
    }

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const out_height = (input_shape[2] - kernel[0]) / stride[0] + 1;
    const out_width = (input_shape[3] - kernel[1]) / stride[1] + 1;

    // std.log.debug("\n[DEBUG] Calculated output dimensions - height: {}, width: {}", .{ out_height, out_width });
    return [4]usize{ batch_size, channels, out_height, out_width };
}

// POOLING -----------------------------------------------------------------------------------------------------------------------
pub fn pool_tensor(
    comptime T: type,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !Tensor(T) {
    var outputTensorShape = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(outputTensorShape);

    for (0..input.shape.len - 2) |i| {
        outputTensorShape[i] = input.shape[i];
    }
    const height = input.shape.len - 2;
    const width = input.shape.len - 1;

    outputTensorShape[height] = (input.shape[height] - kernel[0] + 1) / stride[0];
    outputTensorShape[width] = (input.shape[width] - kernel[1] + 1) / stride[1];

    var output = try Tensor(T).fromShape(&pkg_allocator, outputTensorShape);
    errdefer output.deinit();

    const location = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(location);
    @memset(location, 0);

    try multidim_pooling(
        T,
        input,
        used_windows,
        &output,
        0,
        location,
        kernel,
        stride,
        poolingType,
    );

    return output;
}

pub fn multidim_pooling(
    comptime T: anytype,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    output: *Tensor(T),
    current_depth: usize,
    location: []usize,
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !void {
    if (current_depth == output.shape.len - 2) {
        const allocator = pkg_allocator;

        var temp_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(temp_location);
        var window_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(window_location);
        var window_values = try allocator.alloc(T, kernel[0] * kernel[1]);
        defer allocator.free(window_values);

        var output_row_counter: usize = 0;
        var output_col_counter: usize = 0;

        @memcpy(temp_location, location);
        @memcpy(window_location, location);

        temp_location[current_depth] = 0;
        temp_location[current_depth + 1] = 0;

        const input_rows = input.shape[current_depth];
        const input_cols = input.shape[current_depth + 1];
        const output_cols = output.shape[current_depth + 1];

        while (temp_location[current_depth] + kernel[0] <= input_rows) : (temp_location[current_depth] += stride[0]) {
            temp_location[current_depth + 1] = 0; // Reset column counter
            while (temp_location[current_depth + 1] + kernel[1] <= input_cols) : (temp_location[current_depth + 1] += stride[1]) {
                const kernel_rows = kernel[0];
                const kernel_cols = kernel[1];
                const w = output_row_counter * output_cols + output_col_counter;

                // Reset window location for each window
                @memcpy(window_location, temp_location);

                var window_idx: usize = 0;
                for (0..kernel_rows) |i| {
                    for (0..kernel_cols) |j| {
                        window_location[current_depth] = temp_location[current_depth] + i;
                        window_location[current_depth + 1] = temp_location[current_depth + 1] + j;
                        window_values[window_idx] = try input.get_at(window_location);
                        window_idx += 1;
                    }
                }

                switch (poolingType) {
                    .Max => {
                        var max = window_values[0];
                        var max_idx: usize = 0;
                        for (window_values, 0..) |val, i| {
                            if (val > max) {
                                max = val;
                                max_idx = i;
                            }
                        }

                        const row = max_idx / kernel_cols;
                        const col = max_idx % kernel_cols;
                        const max_r = temp_location[current_depth] + row;
                        const max_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + max_r * input_cols + max_c] = 1;

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, max);
                    },
                    .Min => {
                        var min = window_values[0];
                        var min_idx: usize = 0;
                        for (window_values, 0..) |val, i| {
                            if (val < min) {
                                min = val;
                                min_idx = i;
                            }
                        }

                        const row = min_idx / kernel_cols;
                        const col = min_idx % kernel_cols;
                        const min_r = temp_location[current_depth] + row;
                        const min_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + min_r * input_cols + min_c] = 1;

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, min);
                    },
                    .Avg => {
                        var sum: T = 0;
                        for (window_values) |val| {
                            sum += val;
                        }
                        const avg = sum / @as(T, @floatFromInt(window_values.len));

                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, avg);

                        // Mark all values in window as used for average pooling
                        for (0..kernel_rows) |i| {
                            for (0..kernel_cols) |j| {
                                const avg_r = temp_location[current_depth] + i;
                                const avg_c = temp_location[current_depth + 1] + j;
                                used_windows.data[w * (input_rows * input_cols) + avg_r * input_cols + avg_c] = 1;
                            }
                        }
                    },
                }

                output_col_counter += 1;
            }
            output_row_counter += 1;
            output_col_counter = 0;
        }
    } else {
        for (0..output.shape[current_depth]) |element_at_current_depth| {
            location[current_depth] = element_at_current_depth;
            try multidim_pooling(
                T,
                input,
                used_windows,
                output,
                current_depth + 1,
                location,
                kernel,
                stride,
                poolingType,
            );
        }
    }
}

/// Performs pooling operation on a 4D tensor (batch, channels, height, width)
pub fn pool_forward(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, poolingType: PoolingType) !struct { output: Tensor(T), used_input: Tensor(u8) } {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const input_rows = input.shape[2];
    const input_cols = input.shape[3];

    // Validate input dimensions
    if (input.shape.len != 4) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Validate kernel and stride
    if (kernel[0] == 0 or kernel[1] == 0 or stride[0] == 0 or stride[1] == 0) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output dimensions
    const out_rows = (input_rows -% kernel[0]) / stride[0] +% 1;
    const out_cols = (input_cols -% kernel[1]) / stride[1] +% 1;

    // Validate output dimensions
    if (out_rows == 0 or out_cols == 0 or out_rows > input_rows or out_cols > input_cols) {
        return TensorMathError.InvalidDimensions;
    }

    // Allocate output tensor
    var output_shape = [_]usize{ batch_size, channels, out_rows, out_cols };
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Allocate used_input tensor
    var used_input_shape = [_]usize{ batch_size, channels, out_rows * out_cols, input_rows, input_cols };
    var used_input = try Tensor(u8).fromShape(&pkg_allocator, &used_input_shape);
    errdefer used_input.deinit();
    @memset(used_input.data, 0);

    // Main pooling loop
    for (0..batch_size) |b| {
        for (0..channels) |c| {
            const batch_offset = b * channels * input_rows * input_cols;
            const channel_offset = c * input_rows * input_cols;
            const out_batch_offset = b * channels * out_rows * out_cols;
            const out_channel_offset = c * out_rows * out_cols;

            for (0..out_rows) |out_r| {
                const r_start = out_r * stride[0];
                if (r_start >= input_rows) continue;

                for (0..out_cols) |out_c| {
                    const c_start = out_c * stride[1];
                    if (c_start >= input_cols) continue;

                    const window_index = out_r * out_cols + out_c;
                    const out_idx = out_batch_offset + out_channel_offset + out_r * out_cols + out_c;

                    switch (poolingType) {
                        .Max => {
                            var max_value: T = -std.math.floatMax(T);
                            var max_pos: usize = 0;
                            var found_valid = false;

                            // Scan kernel window
                            for (0..kernel[0]) |kr| {
                                const in_r = r_start + kr;
                                if (in_r >= input_rows) break;

                                for (0..kernel[1]) |kc| {
                                    const in_c = c_start + kc;
                                    if (in_c >= input_cols) break;

                                    const idx = batch_offset + channel_offset + in_r * input_cols + in_c;
                                    if (idx >= input.data.len) break;

                                    const val = input.data[idx];
                                    if (!found_valid or val > max_value) {
                                        max_value = val;
                                        max_pos = in_r * input_cols + in_c;
                                        found_valid = true;
                                    }
                                }
                            }

                            // Set output value
                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = if (found_valid) max_value else 0;

                            // Mark used input
                            if (found_valid) {
                                const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                    c * out_rows * out_cols * input_rows * input_cols +
                                    window_index * input_rows * input_cols + max_pos;
                                if (used_idx >= used_input.data.len) {
                                    output.deinit();
                                    used_input.deinit();
                                    return TensorMathError.InvalidDimensions;
                                }
                                used_input.data[used_idx] = 1;
                            }
                        },
                        .Min => {
                            var min_value: T = if (T == f64) std.math.inf(f64) else std.math.floatMax(T);
                            var min_pos: usize = 0;
                            var found_valid = false;

                            for (0..kernel[0]) |kr| {
                                for (0..kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;

                                    if (in_r < input_rows and in_c < input_cols) {
                                        const idx = b * channels * input_rows * input_cols +
                                            c * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        const val = input.data[idx];
                                        if (!found_valid or val < min_value) {
                                            min_value = val;
                                            min_pos = in_r * input_cols + in_c;
                                            found_valid = true;
                                        }
                                    }
                                }
                            }

                            if (!found_valid) {
                                min_value = 0;
                            }

                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = min_value;

                            if (found_valid) {
                                const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                    c * out_rows * out_cols * input_rows * input_cols +
                                    window_index * input_rows * input_cols + min_pos;
                                if (used_idx >= used_input.data.len) {
                                    output.deinit();
                                    used_input.deinit();
                                    return TensorMathError.InvalidDimensions;
                                }
                                used_input.data[used_idx] = 1;
                            }
                        },
                        .Avg => {
                            var sum: T = 0;
                            var count: usize = 0;

                            for (0..kernel[0]) |kr| {
                                for (0..kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;

                                    if (in_r < input_rows and in_c < input_cols) {
                                        const idx = b * channels * input_rows * input_cols +
                                            c * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        sum += input.data[idx];
                                        count += 1;

                                        const used_idx = b * channels * out_rows * out_cols * input_rows * input_cols +
                                            c * out_rows * out_cols * input_rows * input_cols +
                                            window_index * input_rows * input_cols +
                                            in_r * input_cols + in_c;
                                        if (used_idx >= used_input.data.len) {
                                            output.deinit();
                                            used_input.deinit();
                                            return TensorMathError.InvalidDimensions;
                                        }
                                        used_input.data[used_idx] = 1;
                                    }
                                }
                            }
                            const avg = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
                            if (out_idx >= output.data.len) {
                                output.deinit();
                                used_input.deinit();
                                return TensorMathError.InvalidDimensions;
                            }
                            output.data[out_idx] = avg;
                        },
                    }
                }
            }
        }
    }

    return .{ .output = output, .used_input = used_input };
}
