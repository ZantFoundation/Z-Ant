const std = @import("std");

const ZantStepOptions = @import("zantStepOptions.zig").ZantStepOptions;
const ZantOptions = @import("zantOptions.zig").ZantOptions;

pub const ZantModules = struct {
    zant_mod: *std.Build.Module,
    onnx_mod: *std.Build.Module,
    zant_utils_mod: *std.Build.Module,
    IR_zant_mod: *std.Build.Module,
    codegen_mod: *std.Build.Module,

    pub fn init(b: *std.Build, zantStepOptions: ZantStepOptions) !ZantModules {
        // --- onnx module (fully independent, no external deps) ---
        const onnx_mod = b.createModule(.{ .root_source_file = b.path("src/onnx/onnx.zig") });

        // --- zant_utils module (allocator, type_converter; depends on: build_options) ---
        const zant_utils_mod = b.createModule(.{ .root_source_file = b.path("src/utils/utils.zig") });
        zant_utils_mod.addOptions("build_options", zantStepOptions.build_step_option);

        // --- IR_zant module (depends on: onnx, zant_utils) ---
        const IR_zant_mod = b.createModule(.{ .root_source_file = b.path("src/IR_zant/IR_zant.zig") });
        IR_zant_mod.addImport("onnx", onnx_mod);
        IR_zant_mod.addImport("zant_utils", zant_utils_mod);
        IR_zant_mod.addImport("IR_zant", IR_zant_mod); // self-import for internal files

        // --- codegen module (depends on: IR_zant, onnx, zant_utils, codegen_options, build_options) ---
        const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/codegen/codegen.zig") });
        codegen_mod.addImport("IR_zant", IR_zant_mod);
        codegen_mod.addImport("onnx", onnx_mod);
        codegen_mod.addImport("zant_utils", zant_utils_mod);
        codegen_mod.addOptions("codegen_options", zantStepOptions.codegen_step_option);
        codegen_mod.addOptions("build_options", zantStepOptions.build_step_option);

        // --- zant_mod (top-level entry for mains and tests) ---
        const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
        zant_mod.addOptions("build_options", zantStepOptions.build_step_option);
        zant_mod.addImport("zant", zant_mod); // self-import for backward compat in generated code
        zant_mod.addImport("onnx", onnx_mod);
        zant_mod.addImport("zant_utils", zant_utils_mod);
        zant_mod.addImport("IR_zant", IR_zant_mod);
        zant_mod.addImport("codegen", codegen_mod);

        return ZantModules{
            .zant_mod = zant_mod,
            .onnx_mod = onnx_mod,
            .zant_utils_mod = zant_utils_mod,
            .IR_zant_mod = IR_zant_mod,
            .codegen_mod = codegen_mod,
        };
    }
};
