const zant = @import("../../../zant.zig");
const build_options = @import("build_options");

const Tensor = zant.core.tensor.Tensor;
const TensorModule = zant.core.tensor;

// Add proper flag
const targetIsCortex: bool = true;

pub fn cmsisUsed() bool {
    return (@hasDecl(build_options, "enable_cmsis") and build_options.enable_cmsis and targetIsCortex);
}
