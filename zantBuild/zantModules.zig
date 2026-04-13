const std = @import("std");

const ZantStepOptions = @import("zantStepOptions.zig").ZantStepOptions;
const ZantOptions = @import("zantOptions.zig").ZantOptions;

pub const ZantModules = struct {
    zant_utils_mod: *std.Build.Module,
    IR_zant_mod: *std.Build.Module,
    codegen_mod: *std.Build.Module,

    pub fn init(b: *std.Build, zantStepOptions: ZantStepOptions) !ZantModules {
        // --- zant_utils module (allocator; depends on: build_options) ---
        const zant_utils_mod = b.createModule(.{ .root_source_file = b.path("src/utils/utils.zig") });
        zant_utils_mod.addOptions("build_options", zantStepOptions.build_step_option);

        // --- IR_zant module (depends on: zant_utils) ---
        // onnx is imported via relative path from IR_zant/onnx.zig
        const IR_zant_mod = b.createModule(.{ .root_source_file = b.path("src/codegen/IR_zant.zig") });
        IR_zant_mod.addImport("zant_utils", zant_utils_mod);
        IR_zant_mod.addImport("IR_zant", IR_zant_mod); // self-import for internal files

        // --- codegen module (depends on: IR_zant, zant_utils, codegen_options, build_options) ---
        const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/codegen.zig") });
        codegen_mod.addImport("IR_zant", IR_zant_mod);
        codegen_mod.addImport("zant_utils", zant_utils_mod);
        codegen_mod.addImport("codegen", codegen_mod); // self-import for internal files
        codegen_mod.addOptions("codegen_options", zantStepOptions.codegen_step_option);
        codegen_mod.addOptions("build_options", zantStepOptions.build_step_option);

        return ZantModules{
            .zant_utils_mod = zant_utils_mod,
            .IR_zant_mod = IR_zant_mod,
            .codegen_mod = codegen_mod,
        };
    }
};
