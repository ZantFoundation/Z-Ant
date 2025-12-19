const std = @import("std");
const zant = @import("../../../../zant.zig");

const c = @cImport({
    @cInclude("arm_nnfunctions.h");
});

const SCALE_SHIFT: u5 = 16;

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const qlinearconvUtils = @import("../../TensorMath/tensor_math_utils/qlinearconv_utils.zig");

/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
pub inline fn qlinearconvWithTranspose(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    _x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    _w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
    output: *Tensor(InputType),
    _y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    _: []const u8,
) !void {
    // Suppress unused parameter warnings
    // w_zero_point_any is actually used later when packing weights (readPerChannelZP)

    // ========== VALIDATE INPUTS ==========
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    // Extract dimensions
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    const in_height = x.shape[2];
    const in_width = x.shape[3];
    const out_channels = w.shape[0];
    const weight_in_channels = w.shape[1];
    const kernel_height = w.shape[2];
    const kernel_width = w.shape[3];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h = pads_arr[0];
    const pad_w = pads_arr[1];

    if (group_val == 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * group_val != in_channels or out_channels % group_val != 0) {
        return TensorMathError.InvalidDimensions;
    }

    const group_in_channels = weight_in_channels;
    const group_out_channels = out_channels / group_val;

    // DEBUG: Print tensor dimensions
    // // std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_channels, in_height, in_width });
    // // std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, in_channels, kernel_height, kernel_width });
    // // std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_channels, out_height, out_width });
    // // std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // Extract zero points
    const input_zero_point = qlinearconvUtils.readScalarZP(InputType, x_zero_point);
    const output_zero_point = qlinearconvUtils.readScalarZP(InputType, y_zero_point);

    // DEBUG: Print zero points
    // // std.debug.print("CMSIS DEBUG: input_zero_point: {}, output_zero_point: {}\n", .{ input_zero_point, output_zero_point });

    // Helper functions for zero point and scale extraction
    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    // Extract quantization parameters
    const x_scale_val = asF32(ScaleType, _x_scale.data[0]);
    const y_scale_val = asF32(ScaleType, _y_scale.data[0]);
    const w_scale_data = _w_scale.data;
    const has_per_channel_w_scale = w_scale_data.len == out_channels;

    const multipliers_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(multipliers_buf);
    const shifts_buf = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(shifts_buf);

    for (0..out_channels) |ch| {
        const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
        const scale_ratio = (x_scale_val * w_scale_val) / y_scale_val;
        qlinearconvUtils.quantizeMultiplier(scale_ratio, &multipliers_buf[ch], &shifts_buf[ch]);
    }

    // Now implementing the actual CMSIS-NN convolution with proper u8 to i8 conversion

    // ========== SETUP CMSIS DIMENSIONS ==========
    var input_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(in_channels) };
    var output_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(out_channels) };
    var bias_dims: c.cmsis_nn_dims = .{ .n = 1, .h = 1, .w = 1, .c = @intCast(out_channels) };
    var input_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(group_in_channels) };
    var output_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(group_out_channels) };
    var bias_group_dims: c.cmsis_nn_dims = .{ .n = 1, .h = 1, .w = 1, .c = @intCast(group_out_channels) };

    // Proper CMSIS-NN offset calculation and data conversion
    // CMSIS arm_convolve_s8 expects s8 input/output and uses offsets in the same s8 domain.
    // If our tensors are u8, convert data to s8 domain by subtracting 128, and convert offsets accordingly.
    const is_u8_input = InputType == u8;
    const input_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(input_zero_point)) - 128
    else
        @as(i32, @intCast(input_zero_point));
    const output_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(output_zero_point)) - 128
    else
        @as(i32, @intCast(output_zero_point));

    const cmsis_input_offset = -input_zero_point_s8;
    const cmsis_output_offset = output_zero_point_s8;

    var conv_params: c.cmsis_nn_conv_params = .{
        .input_offset = cmsis_input_offset,
        .output_offset = cmsis_output_offset,
        .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
        .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
        .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
        // CMSIS s8 kernels clamp in s8 domain
        .activation = .{ .min = -128, .max = 127 },
    };

    var quant_params: c.cmsis_nn_per_channel_quant_params = .{
        .multiplier = multipliers_buf.ptr,
        .shift = shifts_buf.ptr,
    };

    // Convert bias to i32 format as expected by CMSIS-NN
    var bias_converted: ?[]i32 = null;
    defer if (bias_converted) |buf| pkg_allocator.free(buf);
    var bias_ptr: ?[*]const i32 = null;

    if (bias) |b| {
        const bias_buf = try pkg_allocator.alloc(i32, out_channels);
        const has_per_channel_bias = b.data.len == out_channels;
        var bias_slice = bias_buf;
        for (0..out_channels) |ch| {
            const bias_val = if (has_per_channel_bias) b.data[ch] else b.data[0];

            bias_slice[ch] = @as(i32, @intCast(bias_val));
        }
        bias_converted = bias_slice;
        bias_ptr = @ptrCast(bias_slice.ptr);
    }

    // Pack weights depending on conv kind:
    // - Regular/grouped conv: OHWI (out, kh, kw, in)
    // - Depthwise conv: [1, kh, kw, C_out] per CMSIS DW wrapper
    const per_channel_w_zp = try pkg_allocator.alloc(i32, out_channels);
    defer pkg_allocator.free(per_channel_w_zp);
    for (0..out_channels) |ch| {
        per_channel_w_zp[ch] = qlinearconvUtils.readPerChannelZP(w_zero_point_any, ch, out_channels);
    }

    const total_weights: usize = out_channels * weight_in_channels * kernel_height * kernel_width;
    var w_packed: []i8 = try pkg_allocator.alloc(i8, total_weights);
    defer pkg_allocator.free(w_packed);

    if (group_val == in_channels) {
        // Depthwise: expect C_out = in_channels * channel_multiplier and layout [1, kh, kw, C_out]
        const kernel_size = kernel_height * kernel_width;
        var wp: usize = 0;
        var spatial: usize = 0;
        while (spatial < kernel_size) : (spatial += 1) {
            var src_idx = spatial;
            var m: usize = 0;
            while (m < out_channels) : (m += 1) {
                const w_q_i32 = @as(i32, @intCast(w.data[src_idx]));
                w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - per_channel_w_zp[m]);
                src_idx += kernel_size; // next output channel's element at same spatial pos
                wp += 1;
            }
        }
    } else {
        // Regular and grouped convolution: pack weights in OHWI order (out, h, w, in)
        const kernel_size = kernel_height * kernel_width;
        const channel_stride = kernel_size;
        const output_stride = weight_in_channels * kernel_size;
        var wp: usize = 0;
        for (0..out_channels) |m| {
            const base_idx = m * output_stride;
            const w_zp = per_channel_w_zp[m];
            var spatial: usize = 0;
            while (spatial < kernel_size) : (spatial += 1) {
                var channel_idx = base_idx + spatial;
                var ch: usize = 0;
                while (ch < weight_in_channels) : (ch += 1) {
                    const w_q_i32 = @as(i32, @intCast(w.data[channel_idx]));
                    w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
                    channel_idx += channel_stride;
                    wp += 1;
                }
            }
        }
    }

    // Allocate buffer required by CMSIS-NN wrapper (regular or depthwise)
    var filter_dims: c.cmsis_nn_dims = .{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
    var filter_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
    var buffer_size: i32 = 0;
    if (group_val == in_channels) {
        buffer_size = c.arm_depthwise_conv_wrapper_s8_get_buffer_size(&.{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        }, &input_dims, &.{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) }, &output_dims);
    } else if (group_val == 1) {
        buffer_size = c.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    } else {
        buffer_size = c.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_group_dims, &filter_group_dims, &output_group_dims);
    }
    if (buffer_size < 0) buffer_size = 0;
    var dyn_buffer: ?[]u8 = null;
    defer if (dyn_buffer) |buf| pkg_allocator.free(buf);
    var buffer_ptr: ?*anyopaque = null;
    var ctx_size_i32: i32 = 0;
    if (buffer_size > 0) {
        const buf = try pkg_allocator.alloc(u8, @intCast(buffer_size));
        dyn_buffer = buf;
        buffer_ptr = buf.ptr;
        ctx_size_i32 = @intCast(buf.len);
    }

    var ctx: c.cmsis_nn_context = .{ .buf = buffer_ptr, .size = ctx_size_i32 };
    // Wrapper API does not use upscale_dims

    // Prepare input/output pointers in s8 NHWC domain
    var input_converted: ?[]i8 = null;
    var output_converted: ?[]i8 = null;
    var grouped_input: ?[]i8 = null;
    var grouped_output: ?[]i8 = null;
    defer if (input_converted) |buf| pkg_allocator.free(buf);
    defer if (output_converted) |buf| pkg_allocator.free(buf);
    defer if (grouped_input) |buf| pkg_allocator.free(buf);
    defer if (grouped_output) |buf| pkg_allocator.free(buf);

    // Convert input from NCHW (our layout) to NHWC (CMSIS layout) and to s8
    const input_len = x.data.len;
    const input_buf = try pkg_allocator.alloc(i8, input_len);
    input_converted = input_buf;
    {
        const spatial_size = in_height * in_width;
        const batch_stride = spatial_size * in_channels;
        const zero_adjust: i32 = if (is_u8_input) 128 else 0;
        var n: usize = 0;
        var src_batch_base: usize = 0;
        var dst_batch_base: usize = 0;
        while (n < batch_size) : (n += 1) {
            var pixel: usize = 0;
            while (pixel < spatial_size) : (pixel += 1) {
                var src_idx = src_batch_base + pixel;
                var dst_idx = dst_batch_base + pixel * in_channels;
                var ch: usize = 0;
                while (ch < in_channels) : (ch += 1) {
                    const raw = @as(i32, @intCast(x.data[src_idx]));
                    input_buf[dst_idx] = @as(i8, @intCast(raw - zero_adjust));
                    src_idx += spatial_size;
                    dst_idx += 1;
                }
            }
            src_batch_base += batch_stride;
            dst_batch_base += batch_stride;
        }
    }
    const input_ptr_s8: [*]const i8 = input_buf.ptr;

    // Always use a temporary NHWC s8 buffer for output (convert+reorder back after)
    const output_len = output.data.len;
    const output_buf = try pkg_allocator.alloc(i8, output_len);
    output_converted = output_buf;
    const output_ptr_s8: [*]i8 = output_buf.ptr;

    if (group_val != 1 and group_val != in_channels) {
        const grouped_input_len = batch_size * in_height * in_width * group_in_channels;
        const grouped_output_len = batch_size * out_height * out_width * group_out_channels;
        grouped_input = try pkg_allocator.alloc(i8, grouped_input_len);
        grouped_output = try pkg_allocator.alloc(i8, grouped_output_len);
    }

    // Call CMSIS-NN wrapper (regular or depthwise)
    var status: c.arm_cmsis_nn_status = c.ARM_CMSIS_NN_SUCCESS;
    if (group_val == in_channels) {
        var dw_params: c.cmsis_nn_dw_conv_params = .{
            .input_offset = cmsis_input_offset,
            .output_offset = cmsis_output_offset,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        };
        // Depthwise expects filter dims [1, kh, kw, C_out]
        var dw_filter_dims: c.cmsis_nn_dims = .{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) };
        status = c.arm_depthwise_conv_wrapper_s8(
            &ctx,
            &dw_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &dw_filter_dims,
            w_packed.ptr,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else if (group_val == 1) {
        status = c.arm_convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &filter_dims,
            w_packed.ptr,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else {
        const grouped_in_buf = grouped_input.?;
        const grouped_out_buf = grouped_output.?;
        const grouped_in_ptr: [*]const i8 = grouped_in_buf.ptr;
        const grouped_out_ptr: [*]i8 = grouped_out_buf.ptr;
        const total_input_pixels_group = batch_size * in_height * in_width;
        const total_output_pixels_group = batch_size * out_height * out_width;
        var g: usize = 0;
        while (g < group_val) : (g += 1) {
            const channel_offset_in = g * group_in_channels;
            const channel_offset_out = g * group_out_channels;
            for (0..total_input_pixels_group) |idx| {
                const src_base = idx * in_channels + channel_offset_in;
                const dst_base = idx * group_in_channels;
                std.mem.copyForwards(i8, grouped_in_buf[dst_base .. dst_base + group_in_channels], input_buf[src_base .. src_base + group_in_channels]);
            }

            const group_channel_offset = g * group_out_channels;
            var group_quant_params: c.cmsis_nn_per_channel_quant_params = .{
                .multiplier = multipliers_buf[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
                .shift = shifts_buf[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
            };
            const weights_offset = g * group_out_channels * group_in_channels * kernel_height * kernel_width;
            const group_weights_ptr = w_packed.ptr + weights_offset;
            const bias_group_ptr = if (bias_ptr) |ptr| ptr + group_channel_offset else null;

            status = c.arm_convolve_wrapper_s8(
                &ctx,
                &conv_params,
                &group_quant_params,
                &input_group_dims,
                grouped_in_ptr,
                &filter_group_dims,
                group_weights_ptr,
                &bias_group_dims,
                if (bias_group_ptr) |ptr| @ptrCast(ptr) else null,
                &output_group_dims,
                grouped_out_ptr,
            );
            if (status != c.ARM_CMSIS_NN_SUCCESS) {
                break;
            }

            for (0..total_output_pixels_group) |idx| {
                const src_base = idx * group_out_channels;
                const dst_base = idx * out_channels + channel_offset_out;
                std.mem.copyForwards(i8, output_buf[dst_base .. dst_base + group_out_channels], grouped_out_buf[src_base .. src_base + group_out_channels]);
            }
        }
    }

    if (status != c.ARM_CMSIS_NN_SUCCESS) {
        return TensorMathError.UnexpectedError;
    }

    // Reorder output from NHWC back to NCHW and convert s8 -> u8 if needed
    {
        const buf = output_converted.?;
        const spatial_size = out_height * out_width;
        const batch_stride = spatial_size * out_channels;
        const zero_restore: i32 = if (is_u8_input) 128 else 0;
        var n: usize = 0;
        var src_batch_base: usize = 0;
        var dst_batch_base: usize = 0;
        while (n < batch_size) : (n += 1) {
            var pixel: usize = 0;
            while (pixel < spatial_size) : (pixel += 1) {
                var src_idx = src_batch_base + pixel * out_channels;
                var dst_idx = dst_batch_base + pixel;
                var ch: usize = 0;
                while (ch < out_channels) : (ch += 1) {
                    const v_i32 = @as(i32, @intCast(buf[src_idx]));
                    const adjusted = v_i32 + zero_restore;
                    if (is_u8_input) {
                        output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
                    } else {
                        output.data[dst_idx] = @as(InputType, @intCast(adjusted));
                    }
                    src_idx += 1;
                    dst_idx += spatial_size;
                }
            }
            src_batch_base += batch_stride;
            dst_batch_base += batch_stride;
        }
    }
}

/// Optimized CMSIS-NN Wrapper
/// Key improvement: Uses Arena allocator to eliminate heap fragmentation and overhead
/// Checks for i8/i32 types to enable Zero-Copy execution where possible.
pub inline fn qlinearconv(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    _x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    _w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
    output: *Tensor(InputType),
    _y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    _: []const u8,
) !void {
    var arena = std.heap.ArenaAllocator.init(pkg_allocator);
    defer arena.deinit();
    const scratch_alloc = arena.allocator();

    // ========== VALIDATE INPUTS ==========
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    // NHWC Layout Extraction
    const batch_size = x.shape[0];
    const in_height = x.shape[1];
    const in_width = x.shape[2];
    const in_channels = x.shape[3];

    // Weights assumed OHWI based on previous code context
    const out_channels = w.shape[0];
    const kernel_height = w.shape[1];
    const kernel_width = w.shape[2];
    const weight_in_channels = w.shape[3];

    const out_height = output.shape[1];
    const out_width = output.shape[2];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h = pads_arr[0];
    const pad_w = pads_arr[1];

    if (group_val == 0 or weight_in_channels * group_val != in_channels or out_channels % group_val != 0) {
        return TensorMathError.InvalidDimensions;
    }

    const group_in_channels = weight_in_channels;
    const group_out_channels = out_channels / group_val;

    // --- Quantization Setup (Using Arena) ---
    const multipliers_buf = try scratch_alloc.alloc(i32, out_channels);
    const shifts_buf = try scratch_alloc.alloc(i32, out_channels);

    // Helper for float casting
    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    const x_scale_val = asF32(ScaleType, _x_scale.data[0]);
    const y_scale_val = asF32(ScaleType, _y_scale.data[0]);
    const w_scale_data = _w_scale.data;
    const has_per_channel_w_scale = w_scale_data.len == out_channels;

    for (0..out_channels) |ch| {
        const w_scale_val = asF32(ScaleType, w_scale_data[if (has_per_channel_w_scale) ch else 0]);
        const scale_ratio = (x_scale_val * w_scale_val) / y_scale_val;
        qlinearconvUtils.quantizeMultiplier(scale_ratio, &multipliers_buf[ch], &shifts_buf[ch]);
    }

    // --- Zero Point & Bias Setup ---
    const input_zero_point = qlinearconvUtils.readScalarZP(InputType, x_zero_point);
    const output_zero_point = qlinearconvUtils.readScalarZP(InputType, y_zero_point);
    const is_u8_input = InputType == u8;

    // CMSIS requires s8 offsets.
    // If input is u8 (0..255), we map it to s8 (-128..127) by subtracting 128.
    // The ZeroPoint must shift accordingly.
    const input_zero_point_s8: i32 = if (is_u8_input) @as(i32, @intCast(input_zero_point)) - 128 else @as(i32, @intCast(input_zero_point));
    const output_zero_point_s8: i32 = if (is_u8_input) @as(i32, @intCast(output_zero_point)) - 128 else @as(i32, @intCast(output_zero_point));
    const cmsis_input_offset = -input_zero_point_s8;
    const cmsis_output_offset = output_zero_point_s8;

    // OPTIMIZATION: Zero-Copy Bias
    // If bias is present and already i32, we pass the pointer directly.
    var bias_ptr: ?[*]const i32 = null;
    if (bias) |b| {
        if (BiasType == i32) {
            // Zero Copy path
            bias_ptr = @ptrCast(b.data.ptr);
        } else {
            // Conversion path (only if types differ)
            const bias_buf = try scratch_alloc.alloc(i32, out_channels);
            const has_per_channel_bias = b.data.len == out_channels;
            for (0..out_channels) |ch| {
                const val = if (has_per_channel_bias) b.data[ch] else b.data[0];
                bias_buf[ch] = @as(i32, @intCast(val));
            }
            bias_ptr = bias_buf.ptr;
        }
    }

    // --- Weight Packing ---
    // Optimization: If WeightType is i8 and no repacking needed (and ZP is 0), strictly we could zero-copy.
    // However, CMSIS often requires reordering for Depthwise [1, H, W, O] vs [O, H, W, I].
    // We stick to the arena allocation here, which is much faster than heap alloc

    // Read Weight ZPs
    const per_channel_w_zp = try scratch_alloc.alloc(i32, out_channels);
    var all_w_zp_zero = true;
    for (0..out_channels) |ch| {
        const val = qlinearconvUtils.readPerChannelZP(w_zero_point_any, ch, out_channels);
        per_channel_w_zp[ch] = val;
        if (val != 0) all_w_zp_zero = false;
    }

    const total_weights: usize = out_channels * weight_in_channels * kernel_height * kernel_width;
    var w_packed_ptr: [*]const i8 = undefined;

    const can_zero_copy_weights = (WeightType == i8) and all_w_zp_zero and (group_val != in_channels);
    if (can_zero_copy_weights) {
        w_packed_ptr = @ptrCast(w.data.ptr);
    } else {
        const w_packed = try scratch_alloc.alloc(i8, total_weights);

        if (group_val == in_channels) {
            // CMSIS DW expects: [1, H, W, Out_Channels]
            var wp: usize = 0;
            for (0..kernel_height) |kh| {
                for (0..kernel_width) |kw| {
                    for (0..out_channels) |oc| { // Output channel is now the inner loop
                        const w_zp = per_channel_w_zp[oc];
                        // Assuming original w is [O, H, W, I] where I=1
                        const idx = oc * (kernel_height * kernel_width) + (kh * kernel_width) + kw;

                        const w_q_i32 = @as(i32, @intCast(w.data[idx]));
                        w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
                        wp += 1;
                    }
                }
            }
        } else {
            // Standard Conv: (w - zp)
            var wp: usize = 0;
            for (0..out_channels) |oc| {
                const w_zp = per_channel_w_zp[oc];
                // Linear iteration if standard layout matches
                const block_size = kernel_height * kernel_width * weight_in_channels;
                const start_idx = oc * block_size;

                // Vectorization hint: This inner loop could be @Vector optimized if strides allow
                for (0..block_size) |i| {
                    const w_q_i32 = @as(i32, @intCast(w.data[start_idx + i]));
                    w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
                    wp += 1;
                }
            }
        }
        w_packed_ptr = w_packed.ptr;
    }

    // --- CMSIS Context & Structs ---
    // [Dims structs same as before]
    var input_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(in_channels) };
    var output_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(out_channels) };
    var bias_dims: c.cmsis_nn_dims = .{ .n = 1, .h = 1, .w = 1, .c = @intCast(out_channels) };

    // [Conv Params same as before]
    var conv_params: c.cmsis_nn_conv_params = .{
        .input_offset = cmsis_input_offset,
        .output_offset = cmsis_output_offset,
        .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
        .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
        .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
        .activation = .{ .min = -128, .max = 127 },
    };
    var quant_params: c.cmsis_nn_per_channel_quant_params = .{ .multiplier = multipliers_buf.ptr, .shift = shifts_buf.ptr };

    // --- Buffer Size Calculation ---
    var buffer_size: i32 = 0;
    // (Logic for buffer_size calc same as original)
    if (group_val == in_channels) {
        // Depthwise logic
        buffer_size = c.arm_depthwise_conv_wrapper_s8_get_buffer_size(&.{ .input_offset = cmsis_input_offset, .output_offset = cmsis_output_offset, .ch_mult = @intCast(group_out_channels), .stride = conv_params.stride, .padding = conv_params.padding, .dilation = conv_params.dilation, .activation = conv_params.activation }, &input_dims, &.{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) }, &output_dims);
    } else if (group_val == 1) {
        const filter_dims: c.cmsis_nn_dims = .{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
        buffer_size = c.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    } else {
        // Grouped
        const input_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(group_in_channels) };
        const filter_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
        const output_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(group_out_channels) };
        buffer_size = c.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_group_dims, &filter_group_dims, &output_group_dims);
    }

    // Alloc Scratch Buffer from Arena
    var ctx: c.cmsis_nn_context = .{ .buf = null, .size = 0 };
    if (buffer_size > 0) {
        const buf = try scratch_alloc.alloc(u8, @intCast(buffer_size));
        ctx.buf = buf.ptr;
        ctx.size = @intCast(buffer_size);
    }

    // --- OPTIMIZATION: Vectorized Input Conversion ---
    var input_ptr_s8: [*]const i8 = undefined;
    if (InputType == i8) {
        input_ptr_s8 = @ptrCast(x.data.ptr);
    } else {
        const count = x.data.len;
        const input_buf = try scratch_alloc.alloc(i8, count);

        // Vector width (128-bit = 16 bytes is safe for most embedded)
        const vec_len = 16;
        const VecU8 = @Vector(vec_len, u8);
        const VecI8 = @Vector(vec_len, i8);

        // 128 is 0x80. In u8, x - 128 is equivalent to x + 128 (wrappinng)
        const offset_vec: VecU8 = @splat(128);

        var i: usize = 0;
        // Vector Loop
        while (i + vec_len <= count) : (i += vec_len) {
            const chunk: VecU8 = x.data[i..][0..vec_len].*;
            // Wrapping sub: 0 - 128 = 128 (-128 in i8). 255 - 128 = 127.
            const res = chunk -% offset_vec;
            const res_i8: VecI8 = @bitCast(res);
            input_buf[i..][0..vec_len].* = res_i8;
        }
        // Scalar Cleanup for remaining items
        while (i < count) : (i += 1) {
            input_buf[i] = @as(i8, @intCast(@as(i32, x.data[i]) - 128));
        }

        input_ptr_s8 = input_buf.ptr;
    }

    // --- Output Buffer ---
    // CMSIS requires an s8 output buffer.
    // If our OutputType is i8, we can write directly to the tensor
    var output_ptr_s8: [*]i8 = undefined;
    const can_zero_copy_output = (InputType == i8);

    if (can_zero_copy_output) {
        output_ptr_s8 = @ptrCast(output.data.ptr);
    } else {
        const output_buf = try scratch_alloc.alloc(i8, output.data.len);
        output_ptr_s8 = output_buf.ptr;
    }

    // --- EXECUTION ---
    var status: c.arm_cmsis_nn_status = c.ARM_CMSIS_NN_SUCCESS;

    if (group_val == in_channels) {
        // DepthWise Conv
        var dw_filter_dims: c.cmsis_nn_dims = .{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) };
        var dw_params: c.cmsis_nn_dw_conv_params = .{ .input_offset = cmsis_input_offset, .output_offset = cmsis_output_offset, .ch_mult = @intCast(group_out_channels), .stride = conv_params.stride, .padding = conv_params.padding, .dilation = conv_params.dilation, .activation = conv_params.activation };
        status = c.arm_depthwise_conv_wrapper_s8(&ctx, &dw_params, &quant_params, &input_dims, input_ptr_s8, &dw_filter_dims, w_packed_ptr, &bias_dims, bias_ptr, &output_dims, output_ptr_s8);
    } else if (group_val == 1) {
        // Standard Conv
        var filter_dims: c.cmsis_nn_dims = .{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
        status = c.arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, &input_dims, input_ptr_s8, &filter_dims, w_packed_ptr, &bias_dims, bias_ptr, &output_dims, output_ptr_s8);
    } else {
        const grouped_input = try scratch_alloc.alloc(i8, batch_size * in_height * in_width * group_in_channels);
        const grouped_output = try scratch_alloc.alloc(i8, batch_size * out_height * out_width * group_out_channels);

        const total_input_pixels_group = batch_size * in_height * in_width;
        const total_output_pixels_group = batch_size * out_height * out_width;
        const input_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(group_in_channels) };
        const filter_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
        const bias_group_dims: c.cmsis_nn_dims = .{ .n = 1, .h = 1, .w = 1, .c = @intCast(group_out_channels) };
        const output_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(group_out_channels) };

        var g: usize = 0;
        while (g < group_val) : (g += 1) {
            const channel_offset_in = g * group_in_channels;
            const channel_offset_out = g * group_out_channels;

            // Copy/Deinterleave inputs
            // Optimization potential: This deinterleaving is slow?????
            for (0..total_input_pixels_group) |idx| {
                const src_base = idx * in_channels + channel_offset_in;
                const dst_base = idx * group_in_channels;
                // Using input_ptr_s8 directly
                @memcpy(grouped_input[dst_base .. dst_base + group_in_channels], input_ptr_s8[src_base .. src_base + group_in_channels]);
            }

            const weights_offset = g * group_out_channels * group_in_channels * kernel_height * kernel_width;
            const group_weights_ptr = w_packed_ptr + weights_offset; // Pointer math on s8
            const bias_group_ptr = if (bias_ptr) |ptr| ptr + channel_offset_out else null;

            var group_quant_params: c.cmsis_nn_per_channel_quant_params = .{
                .multiplier = multipliers_buf[channel_offset_out .. channel_offset_out + group_out_channels].ptr,
                .shift = shifts_buf[channel_offset_out .. channel_offset_out + group_out_channels].ptr,
            };

            status = c.arm_convolve_wrapper_s8(&ctx, &conv_params, &group_quant_params, &input_group_dims, grouped_input.ptr, &filter_group_dims, group_weights_ptr, &bias_group_dims, bias_group_ptr, &output_group_dims, grouped_output.ptr);
            if (status != c.ARM_CMSIS_NN_SUCCESS) break;

            // Interleave Output
            for (0..total_output_pixels_group) |idx| {
                const src_base = idx * group_out_channels;
                const dst_base = idx * out_channels + channel_offset_out;
                @memcpy(output_ptr_s8[dst_base .. dst_base + group_out_channels], grouped_output[src_base .. src_base + group_out_channels]);
            }
        }
    }

    if (status != c.ARM_CMSIS_NN_SUCCESS) return TensorMathError.UnexpectedError;

    // --- Final Copy Back (Only if needed) ---
    // If we used Zero-Copy output (InputType == i8), output.data is already filled
    if (!can_zero_copy_output) {
        // OPTIMIZATION: Vectorized Output Conversion for u8
        if (InputType == u8) {
            const count = output.data.len;
            const vec_len = 16;

            // Define vector types
            const VecI8 = @Vector(vec_len, i8);
            const VecU8 = @Vector(vec_len, u8);

            const offset_vec: VecU8 = @splat(128);

            var i: usize = 0;
            const output_slice_s8 = output_ptr_s8[0..count];

            // Vectorized Loop
            while (i + vec_len <= count) : (i += vec_len) {
                // Load 16 s8 values
                const chunk_s8: VecI8 = output_slice_s8[i..][0..vec_len].*;

                const chunk_u8: VecU8 = @bitCast(chunk_s8);
                const res = chunk_u8 +% offset_vec;

                output.data[i..][0..vec_len].* = res;
            }
            const zero_restore: i32 = 128;
            while (i < count) : (i += 1) {
                const v_i32 = @as(i32, @intCast(output_ptr_s8[i])) + zero_restore;
                output.data[i] = @as(u8, @intCast(std.math.clamp(v_i32, 0, 255)));
            }
        } else {
            // Fallback for non-u8 types (rare)
            const zero_restore: i32 = if (is_u8_input) 128 else 0;
            for (0..output.data.len) |idx| {
                const v_i32 = @as(i32, @intCast(output_ptr_s8[idx])) + zero_restore;
                if (is_u8_input) {
                    output.data[idx] = @as(u8, @intCast(std.math.clamp(v_i32, 0, 255)));
                } else {
                    output.data[idx] = @as(InputType, @intCast(v_i32));
                }
            }
        }
    }
}
