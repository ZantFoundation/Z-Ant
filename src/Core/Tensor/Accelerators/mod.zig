const zant = @import("../../../zant.zig");
const build_options = @import("build_options");
const common = @import("common.zig");

const TensorModule = zant.core.tensor;

pub const AutoPadMode = common.AutoPadMode;
pub const ConvPreparedParams = common.ConvPreparedParams;

fn backendModule() type {
    if (@hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel) {
        return @import("stm32n6.zig");
    }
    return @import("null_accelerator.zig");
}

const Backend = backendModule();

pub fn isStm32n6Enabled() bool {
    return @hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel;
}

pub fn isForceNativeEnabled() bool {
    return @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;
}

pub fn isCmsisRequested() bool {
    return @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
}

pub fn isEthosRequested() bool {
    return @hasDecl(build_options, "stm32n6_use_ethos") and build_options.stm32n6_use_ethos;
}

pub fn canUseCmsisHelium() bool {
    return isStm32n6Enabled() and !isForceNativeEnabled() and isCmsisRequested();
}

pub fn tryConvLean(
    comptime T: type,
    input: *const TensorModule.Tensor(T),
    weight: *const TensorModule.Tensor(T),
    output: *TensorModule.Tensor(T),
    bias: ?[]const T,
    params: ConvPreparedParams,
) !bool {
    if (@hasDecl(Backend, "tryConvLean")) {
        return try Backend.tryConvLean(T, input, weight, output, bias, params);
    }
    return false;
}

pub fn resetTestHooks() void {
    if (@hasDecl(Backend, "resetTestHooks")) {
        Backend.resetTestHooks();
    }
}

pub fn cmsisUsed() bool {
    if (@hasDecl(Backend, "cmsisUsed")) {
        return Backend.cmsisUsed();
    }
    return false;
}

pub fn ethosUsed() bool {
    if (@hasDecl(Backend, "ethosUsed")) {
        return Backend.ethosUsed();
    }
    return false;
}
