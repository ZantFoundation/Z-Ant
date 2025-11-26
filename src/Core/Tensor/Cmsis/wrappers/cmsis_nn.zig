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
    // std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_channels, in_height, in_width });
    // std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, in_channels, kernel_height, kernel_width });
    // std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_channels, out_height, out_width });
    // std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // Extract zero points
    const input_zero_point = qlinearconvUtils.readScalarZP(InputType, x_zero_point);
    const output_zero_point = qlinearconvUtils.readScalarZP(InputType, y_zero_point);

    // DEBUG: Print zero points
    // std.debug.print("CMSIS DEBUG: input_zero_point: {}, output_zero_point: {}\n", .{ input_zero_point, output_zero_point });

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

/// !!! USES CMSIS CONVENTION NHWC
/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
pub inline fn _qlinearconv(
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
    // const batch_size = x.shape[0];
    // const in_channels = x.shape[1];
    // const in_height = x.shape[2];
    // const in_width = x.shape[3];
    // const out_channels = w.shape[0];
    // const weight_in_channels = w.shape[1];
    // const kernel_height = w.shape[2];
    // const kernel_width = w.shape[3];
    // const out_height = output.shape[2];
    // const out_width = output.shape[3];

    // NHWC
    const batch_size = x.shape[0];
    const in_height = x.shape[1];
    const in_width = x.shape[2];
    const in_channels = x.shape[3];

    const out_channels = w.shape[0];
    const kernel_height = w.shape[1];
    const kernel_width = w.shape[2];
    const weight_in_channels = w.shape[3];

    const out_height = output.shape[1];
    const out_width = output.shape[2];

    // ------------------------------------------------------------------------------------
    // ======================  NHWC/OHWI DEBUG INSPECTOR  ============================
    // ------------------------------------------------------------------------------------

    std.debug.print("\n===================== DEBUG: TENSOR LAYOUT CHECK =====================\n", .{});

    // ---- SHAPES ----
    std.debug.print("Input  shape  (expected NHWC):  N={}, H={}, W={}, C={}\n", .{ x.shape[0], x.shape[1], x.shape[2], x.shape[3] });
    std.debug.print("Weights shape (expected OHWI):  O={}, H={}, W={}, I={}\n", .{ w.shape[0], w.shape[1], w.shape[2], w.shape[3] });
    std.debug.print("Output shape (expected NHWC):   N={}, H={}, W={}, C={}\n", .{ output.shape[0], output.shape[1], output.shape[2], output.shape[3] });

    // ---- RAW BUFFER PREVIEW ----
    std.debug.print("\n--- FIRST 5 VALUES OF INPUT BUFFER (RAW) ---\n", .{});
    const n_raw = @min(x.data.len, 5);
    var i: u32 = 0;
    for (x.data[0..n_raw]) |v| {
        std.debug.print("x[{}] = {}\n", .{ i, v });
        i += 1;
    }

    // ---- CHECK WEIGHTS OHWI ----
    std.debug.print("\n--- WEIGHT CHECK (expected OHWI order) ---\n", .{});
    const w_min = @min(w.data.len, 5);
    i = 0;
    for (w.data[0..w_min]) |v| {
        std.debug.print("w[{}] = {}\n", .{ i, v });
        i += 1;
    }
    // ------------------------------------------------------------------------------------

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
    //std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_height, in_width, in_channels });
    //std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, kernel_height, kernel_width, weight_in_channels }); // swap to OHWI
    //std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_height, out_width, out_channels });
    //std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // Extract zero points
    const input_zero_point = qlinearconvUtils.readScalarZP(InputType, x_zero_point);
    const output_zero_point = qlinearconvUtils.readScalarZP(InputType, y_zero_point);

    // DEBUG: Print zero points
    //std.debug.print("CMSIS DEBUG: input_zero_point: {}, output_zero_point: {}\n", .{ input_zero_point, output_zero_point });

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

    // if (group_val == in_channels) {
    //     // Depthwise: expect C_out = in_channels * channel_multiplier and layout [1, kh, kw, C_out]
    //     const kernel_size = kernel_height * kernel_width;
    //     var wp: usize = 0;
    //     var spatial: usize = 0;
    //     while (spatial < kernel_size) : (spatial += 1) {
    //         var src_idx = spatial;
    //         var m: usize = 0;
    //         while (m < out_channels) : (m += 1) {
    //             const w_q_i32 = @as(i32, @intCast(w.data[src_idx]));
    //             w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - per_channel_w_zp[m]);
    //             src_idx += kernel_size; // next output channel's element at same spatial pos
    //             wp += 1;
    //         }
    //     }
    // } else {
    //     // Regular and grouped convolution: pack weights in OHWI order (out, h, w, in)
    //     const kernel_size = kernel_height * kernel_width;
    //     const channel_stride = kernel_size;
    //     const output_stride = weight_in_channels * kernel_size;
    //     var wp: usize = 0;
    //     for (0..out_channels) |m| {
    //         const base_idx = m * output_stride;
    //         const w_zp = per_channel_w_zp[m];
    //         var spatial: usize = 0;
    //         while (spatial < kernel_size) : (spatial += 1) {
    //             var channel_idx = base_idx + spatial;
    //             var ch: usize = 0;
    //             while (ch < weight_in_channels) : (ch += 1) {
    //                 const w_q_i32 = @as(i32, @intCast(w.data[channel_idx]));
    //                 w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
    //                 channel_idx += channel_stride;
    //                 wp += 1;
    //             }
    //         }
    if (group_val == in_channels) {
        var wp: usize = 0;
        // w.data is in [O, H, W, I]
        var oc: usize = 0;
        while (oc < out_channels) : (oc += 1) {
            const w_zp = per_channel_w_zp[oc];
            var kh: usize = 0;
            while (kh < kernel_height) : (kh += 1) {
                var kw: usize = 0;
                while (kw < kernel_width) : (kw += 1) {
                    var ic: usize = 0;
                    while (ic < weight_in_channels) : (ic += 1) {
                        const idx =
                            oc * (kernel_height * kernel_width * weight_in_channels) +
                            kh * (kernel_width * weight_in_channels) +
                            kw * weight_in_channels +
                            ic;
                        const w_q_i32 = @as(i32, @intCast(w.data[idx]));
                        w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
                        wp += 1;
                    }
                }
            }
        }
    } else {
        // Regular and grouped conv: w is already in OHWI, only do (w - zp) e clamp
        var wp: usize = 0;
        var oc: usize = 0;
        while (oc < out_channels) : (oc += 1) {
            const w_zp = per_channel_w_zp[oc];
            var kh: usize = 0;
            while (kh < kernel_height) : (kh += 1) {
                var kw: usize = 0;
                while (kw < kernel_width) : (kw += 1) {
                    var ic: usize = 0;
                    while (ic < weight_in_channels) : (ic += 1) {
                        const idx =
                            oc * (kernel_height * kernel_width * weight_in_channels) +
                            kh * (kernel_width * weight_in_channels) +
                            kw * weight_in_channels +
                            ic;
                        const w_q_i32 = @as(i32, @intCast(w.data[idx]));
                        w_packed[wp] = qlinearconvUtils.clampToI8(w_q_i32 - w_zp);
                        wp += 1;
                    }
                }
            }
        }
    }

    // Allocate buffer required by CMSIS-NN wrapper (regular or depthwise)
    var filter_dims: c.cmsis_nn_dims = .{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
    var filter_group_dims: c.cmsis_nn_dims = .{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };
    //var filter_dims: c.cmsis_nn_dims =
    //   .{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };

    //var filter_group_dims: c.cmsis_nn_dims =
    //   .{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };

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

    const input_len = x.data.len;
    const input_buf = try pkg_allocator.alloc(i8, input_len);
    input_converted = input_buf;
    // {
    //     const spatial_size = in_height * in_width;
    //     const batch_stride = spatial_size * in_channels;
    //     const zero_adjust: i32 = if (is_u8_input) 128 else 0;
    //     var n: usize = 0;
    //     var src_batch_base: usize = 0;
    //     var dst_batch_base: usize = 0;
    //     while (n < batch_size) : (n += 1) {
    //         var pixel: usize = 0;
    //         while (pixel < spatial_size) : (pixel += 1) {
    //             var src_idx = src_batch_base + pixel;
    //             var dst_idx = dst_batch_base + pixel * in_channels;
    //             var ch: usize = 0;
    //             while (ch < in_channels) : (ch += 1) {
    //                 const raw = @as(i32, @intCast(x.data[src_idx]));
    //                 input_buf[dst_idx] = @as(i8, @intCast(raw - zero_adjust));
    //                 src_idx += spatial_size;
    //                 dst_idx += 1;
    //             }
    //         }
    //         src_batch_base += batch_stride;
    //         dst_batch_base += batch_stride;
    //     }
    // }

    // We already have NHWC, just convert from u8 -> s8
    const zero_adjust: i32 = if (is_u8_input) 128 else 0;
    for (0..input_len) |idx| {
        const raw = @as(i32, @intCast(x.data[idx]));
        input_buf[idx] = @as(i8, @intCast(raw - zero_adjust));
    }

    const input_ptr_s8: [*]const i8 = input_buf.ptr;

    // Always use a temporary NHWC s8 buffer for output
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
    // {
    //     const buf = output_converted.?;
    //     const spatial_size = out_height * out_width;
    //     const batch_stride = spatial_size * out_channels;
    //     const zero_restore: i32 = if (is_u8_input) 128 else 0;
    //     var n: usize = 0;
    //     var src_batch_base: usize = 0;
    //     var dst_batch_base: usize = 0;
    //     while (n < batch_size) : (n += 1) {
    //         var pixel: usize = 0;
    //         while (pixel < spatial_size) : (pixel += 1) {
    //             var src_idx = src_batch_base + pixel * out_channels;
    //             var dst_idx = dst_batch_base + pixel;
    //             var ch: usize = 0;
    //             while (ch < out_channels) : (ch += 1) {
    //                 const v_i32 = @as(i32, @intCast(buf[src_idx]));
    //                 const adjusted = v_i32 + zero_restore;
    //                 if (is_u8_input) {
    //                     output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(adjusted, 0, 255)));
    //                 } else {
    //                     output.data[dst_idx] = @as(InputType, @intCast(adjusted));
    //                 }
    //                 src_idx += 1;
    //                 dst_idx += spatial_size;
    //             }
    //         }
    //         src_batch_base += batch_stride;
    //         dst_batch_base += batch_stride;
    //     }
    // }

    // Output is NHWC already; convert i8 → InputType
    const buf = output_converted.?;
    const zero_restore: i32 = if (is_u8_input) 128 else 0;
    for (0..output.data.len) |idx| {
        const v_i32 = @as(i32, @intCast(buf[idx])) + zero_restore;

        if (is_u8_input) {
            output.data[idx] = @as(u8, @intCast(std.math.clamp(v_i32, 0, 255)));
        } else {
            output.data[idx] = @as(InputType, @intCast(v_i32));
        }
    }
}
