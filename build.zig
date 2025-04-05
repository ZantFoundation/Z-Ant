const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

    // Get target and CPU options from command line or use defaults
    const target_str = b.option([]const u8, "target", "Target architecture (e.g., thumb-freestanding)") orelse "native";
    const cpu_str = b.option([]const u8, "cpu", "CPU model (e.g., cortex_m33)");

    const target_query = std.Target.Query.parse(.{
        .arch_os_abi = target_str,
        .cpu_features = cpu_str,
    }) catch |err| {
        std.debug.print("Error parsing target: {}\n", .{err});
        return;
    };

    const target = b.resolveTargetQuery(target_query);
    const optimize = b.standardOptimizeOption(.{});

    // -------------------- Modules creation

    const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
    zant_mod.addOptions("build_options", build_options);

    const codeGen_mod = b.createModule(.{ .root_source_file = b.path("src/CodeGen/codegen.zig") });
    codeGen_mod.addImport("zant", zant_mod);

    //************************************************UNIT TESTS************************************************

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "test_lib",
        .root_source_file = b.path("tests/test_lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define test options
    const test_options = b.addOptions();
    test_options.addOption(bool, "heavy", b.option(bool, "heavy", "Run heavy tests") orelse false);
    unit_tests.root_module.addOptions("test_options", test_options);

    const test_name = b.option([]const u8, "test_name", "specify a test name to run") orelse "";
    test_options.addOption([]const u8, "test_name", test_name);

    unit_tests.root_module.addImport("zant", zant_mod);
    unit_tests.root_module.addImport("codegen", codeGen_mod);

    unit_tests.linkLibC();

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ************************************************CODEGEN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const codeGen_exe = b.addExecutable(.{
        .name = "Codegen",
        .root_source_file = b.path("src/CodeGen/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    codeGen_exe.linkLibC();

    // Add necessary imports for the executable.
    codeGen_exe.root_module.addImport("zant", zant_mod);

    // name of the model
    const model_name_option = b.option([]const u8, "model", "Model name") orelse "mnist-8";

    // Define codegen options
    const codegen_options = b.addOptions(); // Model name option
    codegen_options.addOption([]const u8, "model", model_name_option);
    codegen_options.addOption([]const u8, "user_tests", b.option([]const u8, "user_tests", "User tests path") orelse "");
    codegen_options.addOption(bool, "log", b.option(bool, "log", "Run with log") orelse false);
    codegen_options.addOption([]const u8, "shape", b.option([]const u8, "shape", "Input shape") orelse "");
    codegen_options.addOption([]const u8, "type", b.option([]const u8, "type", "Input type") orelse "f32");
    codegen_options.addOption(bool, "comm", b.option(bool, "comm", "Codegen with comments") orelse false);
    codeGen_exe.root_module.addOptions("codegen_options", codegen_options);

    // Install the executable.
    b.installArtifact(codeGen_exe);

    // Define the run command for the main executable.
    const codegen_cmd = b.addRunArtifact(codeGen_exe);
    if (b.args) |args| {
        codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const codegen_step = b.step("codegen", "code generation");
    codegen_step.dependOn(&codegen_cmd.step);

    // ************************************************ STATIC LIBRARY CREATION ************************************************

    const lib_model_path = std.fmt.allocPrint(b.allocator, "generated/{s}/lib_{s}.zig", .{ model_name_option, model_name_option }) catch |err| {
        std.debug.print("Error allocating model path: {}\n", .{err});
        return;
    };

    const static_lib = b.addStaticLibrary(.{
        .name = "zant",
        .root_source_file = b.path(lib_model_path),
        .target = target,
        .optimize = optimize,
    });
    static_lib.linkLibC();
    static_lib.root_module.addImport("zant", zant_mod);
    static_lib.root_module.addImport("codegen", codeGen_mod);

    const install_lib_step = b.addInstallArtifact(static_lib, .{ .dest_dir = .{ .override = .{ .custom = model_name_option } } });
    const lib_step = b.step("lib", "Compile tensor_math static library");
    lib_step.dependOn(&install_lib_step.step);

    // ************************************************ GENERATED LIBRARY TESTS ************************************************

    // Add test for generated library
    const test_model_path = std.fmt.allocPrint(b.allocator, "generated/{s}/test_{s}.zig", .{ model_name_option, model_name_option }) catch |err| {
        std.debug.print("Error allocating model path: {}\n", .{err});
        return;
    };

    const test_generated_lib = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path(test_model_path),
        .target = target,
        .optimize = optimize,
    });

    test_generated_lib.root_module.addImport("zant", zant_mod);
    test_generated_lib.root_module.addImport("codegen", codeGen_mod);
    test_generated_lib.linkLibC();

    const run_test_generated_lib = b.addRunArtifact(test_generated_lib);
    const test_step_generated_lib = b.step("test-generated-lib", "Run generated library tests");
    test_step_generated_lib.dependOn(&run_test_generated_lib.step);

    // ************************************************ ONEOP CODEGEN ************************************************
    // Setup oneOp codegen

    const oneop_codegen_exe = b.addExecutable(.{
        .name = "oneop_codegen",
        .root_source_file = b.path("tests/CodeGen/oneOpModelGenerator.zig"),
        .target = target,
        .optimize = optimize,
    });

    oneop_codegen_exe.root_module.addImport("zant", zant_mod);
    codeGen_mod.addOptions("codegen_options", codegen_options);
    oneop_codegen_exe.root_module.addImport("codegen", codeGen_mod);
    oneop_codegen_exe.linkLibC();

    const run_oneop_codegen_exe = b.addRunArtifact(oneop_codegen_exe);
    const step_test_oneOp_codegen = b.step("test-codegen-gen", "Run generated library tests");
    step_test_oneOp_codegen.dependOn(&run_oneop_codegen_exe.step);

    // ************************************************
    // Setup test_all_oneOp

    const test_all_oneOp = b.addTest(.{
        .name = "test_all_oneOp",
        .root_source_file = b.path("generated/oneOpModels/test_oneop_models.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_all_oneOp.root_module.addImport("zant", zant_mod);
    codeGen_mod.addOptions("codegen_options", codegen_options);
    test_all_oneOp.root_module.addImport("codegen", codeGen_mod);
    test_all_oneOp.linkLibC();

    const run_test_all_oneOp = b.addRunArtifact(test_all_oneOp);

    // ************************************************
    // Setup test oneop
    // It will run
    // - run_oneop_codegen_exe
    // - run_test_all_oneOp

    const step_test_oneOp = b.step("test-codegen", "Run generated library tests");
    step_test_oneOp.dependOn(&run_test_all_oneOp.step);

    // ************************************************ ONNX PARSER TESTS ************************************************
    // Add test for generated library

    const test_onnx_parser = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path("tests/Onnx/onnx_loader.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_onnx_parser.root_module.addImport("zant", zant_mod);
    test_onnx_parser.linkLibC();

    const run_test_onnx_parser = b.addRunArtifact(test_onnx_parser);
    const step_test_onnx_parser = b.step("onnx-parser", "Run generated library tests");
    step_test_onnx_parser.dependOn(&run_test_onnx_parser.step);
}
