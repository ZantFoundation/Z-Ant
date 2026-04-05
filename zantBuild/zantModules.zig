const std = @import("std");

const ZantStepOptions = @import("zantStepOptions.zig").ZantStepOptions;
const ZantOptions = @import("zantOptions.zig").ZantOptions;

pub const ZantModules = struct {
    zant_mod: *std.Build.Module,
    IR_zant_mod: *std.Build.Module,
    codegen_mod: *std.Build.Module,
    Img2Tens_mod: *std.Build.Module,

    pub fn init(b: *std.Build, zantStepOptions: ZantStepOptions) !ZantModules {
        const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
        zant_mod.addOptions("build_options", zantStepOptions.build_step_option);
        // Allow the zant module to self-import: operator math files pulled in through
        // tensor.zig → zant_math_standard.zig use @import("zant") for Tensor/allocator.
        zant_mod.addImport("zant", zant_mod);

        const IR_zant_mod = b.createModule(.{ .root_source_file = b.path("src/IR_zant/IR_zant.zig") });
        IR_zant_mod.addImport("zant", zant_mod);

        const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/codegen/codegen.zig") });
        codegen_mod.addImport("zant", zant_mod);
        codegen_mod.addImport("IR_zant", IR_zant_mod);
        codegen_mod.addOptions("codegen_options", zantStepOptions.codegen_step_option); //<<--OSS!! it is an option!
        IR_zant_mod.addImport("codegen", codegen_mod);

        const Img2Tens_mod = b.createModule(.{ .root_source_file = b.path("src/ImageToTensor/imageToTensor.zig") });
        Img2Tens_mod.addImport("zant", zant_mod);

        return ZantModules{
            .zant_mod = zant_mod,
            .IR_zant_mod = IR_zant_mod,
            .codegen_mod = codegen_mod,
            .Img2Tens_mod = Img2Tens_mod,
        };
    }
};
