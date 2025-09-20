const builtin = @import("builtin");
const zant = @import("../../../zant.zig");
const common = @import("common.zig");
const build_options = @import("build_options");

const TensorModule = zant.core.tensor;

const force_native = @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;
const use_cmsis = @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
const use_ethos = @hasDecl(build_options, "stm32n6_use_ethos") and build_options.stm32n6_use_ethos;

extern fn zant_stm32n6_conv_f32(
    input_ptr: [*c]const f32,
    input_shape: [*c]const usize,
    weight_ptr: [*c]const f32,
    weight_shape: [*c]const usize,
    output_ptr: [*c]f32,
    output_shape: [*c]const usize,
    bias_ptr: ?*const f32,
    bias_len: usize,
    stride_ptr: [*c]const usize,
    pads_ptr: [*c]const usize,
    dilations_ptr: [*c]const usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
) callconv(.C) bool;

extern fn zant_stm32n6_conv_f32_helium(
    input_ptr: [*c]const f32,
    input_shape: [*c]const usize,
    weight_ptr: [*c]const f32,
    weight_shape: [*c]const usize,
    output_ptr: [*c]f32,
    output_shape: [*c]const usize,
    bias_ptr: ?*const f32,
    bias_len: usize,
    stride_ptr: [*c]const usize,
    pads_ptr: [*c]const usize,
    dilations_ptr: [*c]const usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
) callconv(.C) bool;

extern fn zant_stm32n6_conv_f32_ethos(
    input_ptr: [*c]const f32,
    input_shape: [*c]const usize,
    weight_ptr: [*c]const f32,
    weight_shape: [*c]const usize,
    output_ptr: [*c]f32,
    output_shape: [*c]const usize,
    bias_ptr: ?*const f32,
    bias_len: usize,
    stride_ptr: [*c]const usize,
    pads_ptr: [*c]const usize,
    dilations_ptr: [*c]const usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
) callconv(.C) bool;

extern fn zant_stm32n6_reset_test_state() callconv(.C) void;
extern fn zant_stm32n6_cmsis_was_used() callconv(.C) bool;
extern fn zant_stm32n6_ethos_was_used() callconv(.C) bool;

inline fn archSupported() bool {
    if (force_native) return true;
    return builtin.target.cpu.arch == .thumb or builtin.target.cpu.arch == .thumbeb;
}

pub fn tryConvLean(
    comptime T: type,
    input: *const TensorModule.Tensor(T),
    weight: *const TensorModule.Tensor(T),
    output: *TensorModule.Tensor(T),
    bias: ?[]const T,
    params: common.ConvPreparedParams,
) !bool {
    if (T != f32) {
        return false;
    }

    if (input.shape.len != 4 or weight.shape.len != 4 or output.shape.len != 4) {
        return false;
    }

    if (params.group == 0) {
        return false;
    }

    var input_shape = [_]usize{
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    };

    var weight_shape = [_]usize{
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    };

    var output_shape = [_]usize{
        output.shape[0],
        output.shape[1],
        output.shape[2],
        output.shape[3],
    };

    var stride = params.stride;
    var pads = params.pads;
    var dilations = params.dilations;

    const bias_ptr: ?*const f32 = if (bias) |b| @as(*const f32, @ptrCast(b.ptr)) else null;
    const bias_len: usize = if (bias) |b| b.len else 0;

    if (use_ethos and archSupported()) {
        const ok = zant_stm32n6_conv_f32_ethos(
            @as([*c]const f32, @ptrCast(input.data.ptr)),
            @as([*c]const usize, @ptrCast(input_shape[0..].ptr)),
            @as([*c]const f32, @ptrCast(weight.data.ptr)),
            @as([*c]const usize, @ptrCast(weight_shape[0..].ptr)),
            @as([*c]f32, @ptrCast(output.data.ptr)),
            @as([*c]const usize, @ptrCast(output_shape[0..].ptr)),
            bias_ptr,
            bias_len,
            @as([*c]const usize, @ptrCast(stride[0..].ptr)),
            @as([*c]const usize, @ptrCast(pads[0..].ptr)),
            @as([*c]const usize, @ptrCast(dilations[0..].ptr)),
            params.group,
            params.filters_per_group,
            params.channels_per_group,
        );
        if (ok) return true;
    }

    if (use_cmsis and archSupported()) {
        const ok = zant_stm32n6_conv_f32_helium(
            @as([*c]const f32, @ptrCast(input.data.ptr)),
            @as([*c]const usize, @ptrCast(input_shape[0..].ptr)),
            @as([*c]const f32, @ptrCast(weight.data.ptr)),
            @as([*c]const usize, @ptrCast(weight_shape[0..].ptr)),
            @as([*c]f32, @ptrCast(output.data.ptr)),
            @as([*c]const usize, @ptrCast(output_shape[0..].ptr)),
            bias_ptr,
            bias_len,
            @as([*c]const usize, @ptrCast(stride[0..].ptr)),
            @as([*c]const usize, @ptrCast(pads[0..].ptr)),
            @as([*c]const usize, @ptrCast(dilations[0..].ptr)),
            params.group,
            params.filters_per_group,
            params.channels_per_group,
        );
        if (ok) return true;
    }

    const ok = zant_stm32n6_conv_f32(
        @as([*c]const f32, @ptrCast(input.data.ptr)),
        @as([*c]const usize, @ptrCast(input_shape[0..].ptr)),
        @as([*c]const f32, @ptrCast(weight.data.ptr)),
        @as([*c]const usize, @ptrCast(weight_shape[0..].ptr)),
        @as([*c]f32, @ptrCast(output.data.ptr)),
        @as([*c]const usize, @ptrCast(output_shape[0..].ptr)),
        bias_ptr,
        bias_len,
        @as([*c]const usize, @ptrCast(stride[0..].ptr)),
        @as([*c]const usize, @ptrCast(pads[0..].ptr)),
        @as([*c]const usize, @ptrCast(dilations[0..].ptr)),
        params.group,
        params.filters_per_group,
        params.channels_per_group,
    );

    return ok;
}

pub fn resetTestHooks() void {
    zant_stm32n6_reset_test_state();
}

pub fn cmsisUsed() bool {
    return zant_stm32n6_cmsis_was_used();
}

pub fn ethosUsed() bool {
    return zant_stm32n6_ethos_was_used();
}
