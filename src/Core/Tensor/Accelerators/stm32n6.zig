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

fn OperationDispatch(comptime FnType: type) type {
    return struct {
        reference: FnType,
        cmsis: ?FnType = null,
        ethos: ?FnType = null,
    };
}

fn callOperation(
    comptime FnType: type,
    dispatch: OperationDispatch(FnType),
    args: anytype,
) bool {
    const call_args = args;
    const arch_ok = archSupported();

    if (use_ethos and arch_ok) {
        if (dispatch.ethos) |ethos_fn| {
            if (@call(.auto, ethos_fn, call_args)) {
                return true;
            }
        }
    }

    if (use_cmsis and arch_ok) {
        if (dispatch.cmsis) |cmsis_fn| {
            if (@call(.auto, cmsis_fn, call_args)) {
                return true;
            }
        }
    }

    return @call(.auto, dispatch.reference, call_args);
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

    const c_input = @as([*c]const f32, @ptrCast(input.data.ptr));
    const c_input_shape = @as([*c]const usize, @ptrCast(input_shape[0..].ptr));
    const c_weight = @as([*c]const f32, @ptrCast(weight.data.ptr));
    const c_weight_shape = @as([*c]const usize, @ptrCast(weight_shape[0..].ptr));
    const c_output = @as([*c]f32, @ptrCast(output.data.ptr));
    const c_output_shape = @as([*c]const usize, @ptrCast(output_shape[0..].ptr));
    const c_stride = @as([*c]const usize, @ptrCast(stride[0..].ptr));
    const c_pads = @as([*c]const usize, @ptrCast(pads[0..].ptr));
    const c_dilations = @as([*c]const usize, @ptrCast(dilations[0..].ptr));

    const ConvFn = @TypeOf(zant_stm32n6_conv_f32);
    const dispatch = OperationDispatch(ConvFn){
        .reference = zant_stm32n6_conv_f32,
        .cmsis = zant_stm32n6_conv_f32_helium,
        .ethos = zant_stm32n6_conv_f32_ethos,
    };

    const call_args = .{
        c_input,
        c_input_shape,
        c_weight,
        c_weight_shape,
        c_output,
        c_output_shape,
        bias_ptr,
        bias_len,
        c_stride,
        c_pads,
        c_dilations,
        params.group,
        params.filters_per_group,
        params.channels_per_group,
    };

    return callOperation(ConvFn, dispatch, call_args);
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
