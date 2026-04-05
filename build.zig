const std = @import("std");
const ZantBuild = @import("zantBuild/zantBuild.zig").ZantBuild;

// Global target and optimization
var target: std.Build.ResolvedTarget = undefined;
var optimize: std.builtin.OptimizeMode = undefined;

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {

    // First build the Zant enviroment
    const zantBuild: ZantBuild = ZantBuild.init(b) catch unreachable;
    // ---------------------------------------------------------------------
    // ------------------ Target and Release Optimization ------------------
    // ---------------------------------------------------------------------
    // Get target and CPU options from command line or use defaults
    const target_str = b.option([]const u8, "target", "Target architecture (e.g., thumb-freestanding)") orelse "native";
    const cpu_str = b.option([]const u8, "cpu", "CPU model (e.g., cortex_m33)");

    const target_query = std.Target.Query.parse(.{
        .arch_os_abi = target_str,
        .cpu_features = cpu_str,
    }) catch |err| {
        std.log.scoped(.build).warn("Error parsing target: {}\n", .{err});
        return;
    };
    target = b.resolveTargetQuery(target_query);
    optimize = b.standardOptimizeOption(.{});

    // ************************************************ UNIT TESTS **************************************************
    // $ zig build test --summary all
    unit_test_creation(b, zantBuild);

    // ************************************************ CODEGEN IR LIB_MODEL*****************************************
    // $ zig build lib-gen -Dmodel="myModel" ...
    lib_codegen(b, zantBuild);

    // ************************************************ LIB_MODEL EXECUTABLE ****************************************
    // $ zig build lib-exe -Dmodel="myModel" ...
    lib_exe(b, zantBuild);

    // ************************************************ GENERATED LIBRARY TESTS **************************************
    // $ zig build lib-test -Dmodel="myModel" ...
    lib_test(b, zantBuild);

    // ************************************************ STATIC LIBRARY CREATION **************************************
    // $ zig build lib -Dmodel="myModel" [ -Dtarget=... -Dcpu=... -Doptimize=[ReleaseSmall, ReleaseFast]]
    const static_lib: *std.Build.Step.Compile = lib_creation(b, zantBuild) catch unreachable;

    // ************************************************ ONEOP CODEGEN ************************************************
    // $ zig build op-codegen-gen [ -Dop="OpName" ]
    op_codegen_gen(b, zantBuild);

    // ************************************************ ONEOP TESTING ************************************************
    // $ zig build op-codegen-test [ -Dop="OpName" ]
    op_codegen_test(b, zantBuild);

    // ************************************************ NODE EXTRACTOR GEN *******************************************
    // $ zig build extractor-gen -Dmodel="myModel"
    extractor_gen(b, zantBuild);

    // ************************************************ NODE EXTRACTOR TEST ******************************************
    // $ zig build extractor-test -Dmodel="myModel"
    extractor_test(b, zantBuild);

    // ************************************************ BENCHMARK  ***************************************************
    // $ zig build benchmark
    benchmark_create(b, zantBuild);

    // ************************************************ ONNX PARSER TESTS ********************************************
    // $ zig build onnx-parser
    onnx_parser(b, zantBuild);

    // ************************************************ MAIN EXECUTABLE (for profiling) ******************************
    // $ zig build build-main -Dmodel="my_model"
    build_main(b, zantBuild, static_lib);
}

inline fn unit_test_creation(b: *std.Build, zantBuild: ZantBuild) void {
    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "test_lib",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    unit_tests.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    unit_tests.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    unit_tests.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod);

    unit_tests.linkLibC();

    // Operator math tests: the test index (test_tensor_math.zig) lives with the
    // operator source files and uses @import("zant"). It is added as a named module
    // ("op_tests") so that operator .zig files remain in a single module and don't
    // conflict with the test root module.
    const op_test_mod = b.createModule(.{
        .root_source_file = b.path("src/IR_zant/op_union/test_tensor_math.zig"),
        .target = target,
        .optimize = optimize,
    });
    op_test_mod.addImport("zant", zantBuild.zantModules.zant_mod);
    op_test_mod.addOptions("build_options", zantBuild.zantStepOptions.build_step_option);

    const op_tests = b.addTest(.{
        .name = "test_operators",
        .root_module = op_test_mod,
    });
    op_tests.linkLibC();

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_op_tests = b.addRunArtifact(op_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_op_tests.step);
}

inline fn lib_codegen(b: *std.Build, zantBuild: ZantBuild) void {
    const IR_codeGen_exe = b.addExecutable(.{
        .name = "codegen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    IR_codeGen_exe.linkLibC();

    // Add necessary imports for the executable.
    IR_codeGen_exe.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //<<-- options are inside this module
    IR_codeGen_exe.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    IR_codeGen_exe.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);

    // Define the run command for the main executable.
    const IR_codegen_cmd = b.addRunArtifact(IR_codeGen_exe);
    if (b.args) |args| {
        IR_codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const IR_codegen_step = b.step("lib-gen", "code generation");
    IR_codegen_step.dependOn(&IR_codegen_cmd.step);
}

inline fn lib_exe(b: *std.Build, zantBuild: ZantBuild) void {
    const generated_lib_root = std.fmt.allocPrint(b.allocator, "generated/{s}/lib_{s}.zig", .{ zantBuild.zantOptions.codegen_flags.model_name_option, zantBuild.zantOptions.codegen_flags.model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating generated_lib_root path: {}\n", .{err});
        return;
    };

    const exe_name = std.fmt.allocPrint(b.allocator, "{s}_exe", .{zantBuild.zantOptions.codegen_flags.model_name_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating exe_name: {}\n", .{err});
        return;
    };

    const lib_model_exe = b.addExecutable(.{
        .name = exe_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(generated_lib_root),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add necessary imports for the executable.
    lib_model_exe.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod);
    lib_model_exe.root_module.addImport("zant", zantBuild.zantModules.zant_mod);

    const model_exe_cmd = b.addRunArtifact(lib_model_exe);
    if (b.args) |args| {
        model_exe_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const model_exe_step = b.step("lib-exe", "code generation");
    model_exe_step.dependOn(&model_exe_cmd.step);
}

inline fn lib_test(b: *std.Build, zantBuild: ZantBuild) void {
    //
    // OPTIONS: see codegen_options
    //
    // IMPORTANT: for this YOU need to add a main.zig in generated/your_mode/lib_your_mode.zig !!
    //

    // Add test for generated library
    const test_model_path = std.fmt.allocPrint(b.allocator, "{s}test_{s}.zig", .{ zantBuild.zantOptions.codegen_flags.generated_path_option, zantBuild.zantOptions.codegen_flags.model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating test model path: {}\n", .{err});
        return;
    };

    const test_generated_lib = b.addTest(.{
        .name = "test_generated_lib",
        .root_module = b.createModule(.{
            .root_source_file = b.path(test_model_path),
            .target = target,
            .optimize = .Debug,
        }),
    });

    test_generated_lib.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    test_generated_lib.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod); //codegen
    test_generated_lib.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod);
    test_generated_lib.linkLibC();

    const run_test_generated_lib = b.addRunArtifact(test_generated_lib);
    const test_step_generated_lib = b.step("lib-test", "Run generated library tests");
    test_step_generated_lib.dependOn(&run_test_generated_lib.step);
}

inline fn lib_creation(b: *std.Build, zantBuild: ZantBuild) !*std.Build.Step.Compile {
    const lib_model_path = std.fmt.allocPrint(b.allocator, "{s}lib_{s}.zig", .{ zantBuild.zantOptions.codegen_flags.generated_path_option, zantBuild.zantOptions.codegen_flags.model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating lib model path: {}\n", .{err});
        return err;
    };

    const static_lib: *std.Build.Step.Compile = b.addLibrary(.{
        .name = "zant",
        .root_module = b.createModule(.{
            .root_source_file = b.path(lib_model_path),
            .target = target,
            .optimize = optimize,
        }),
    });

    static_lib.linkLibC();
    static_lib.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    static_lib.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod);

    const output_path = std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ zantBuild.zantOptions.codegen_flags.model_name_option, @tagName(target.result.os.tag) }) catch |err| {
        std.log.scoped(.build).warn("Error allocating old path: {}\n", .{err});
        return err;
    };

    std.debug.print("\n<<<<<<<<{s}", .{output_path});

    const install_lib_step = b.addInstallArtifact(
        static_lib,
        .{
            .dest_dir = .{
                .override = .{
                    .custom = output_path,
                },
            },
        },
    );
    const lib_step = b.step("lib", "Compile tensor_math static library");
    lib_step.dependOn(&install_lib_step.step);

    // Output path for the generated library
    var output_path_option = b.option([]const u8, "output_path", "Output path") orelse "";
    if (output_path_option.len != 0) {
        if (!std.mem.endsWith(u8, output_path_option, "/")) {
            output_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{output_path_option}) catch |err| {
                std.log.scoped(.build).warn("Error normalizing path: {}\n", .{err});
                return err;
            };
        }
        const old_path = std.fmt.allocPrint(b.allocator, "zig-out/{s}/libzant.a", .{output_path}) catch |err| {
            std.log.scoped(.build).warn("Error allocating old path: {}\n", .{err});
            return err;
        };

        const move_step = b.addSystemCommand(&[_][]const u8{
            "mv",
            old_path,
            output_path_option,
        });
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
    }

    return static_lib;
}

inline fn op_codegen_gen(b: *std.Build, zantBuild: ZantBuild) void { // Setup oneOp codegen
    // Remember to launch : python3 tests/CodeGen/Python-ONNX/onnx_gen.py to generate the onnx models
    // see: TESTING OPTIONS

    const oneop_codegen_exe = b.addExecutable(.{
        .name = "oneop_codegen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/CodeGen/oneOpModelGenerator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    oneop_codegen_exe.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    oneop_codegen_exe.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    oneop_codegen_exe.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //codegen
    oneop_codegen_exe.root_module.addOptions("testing_options", zantBuild.zantStepOptions.testing_step_option); //<<--OSS!! it is an option!
    oneop_codegen_exe.linkLibC();

    const run_oneop_codegen_exe = b.addRunArtifact(oneop_codegen_exe);
    const step_test_oneOp_codegen = b.step("op-codegen-gen", "Codegenerate library tests");
    step_test_oneOp_codegen.dependOn(&run_oneop_codegen_exe.step);
}

inline fn op_codegen_test(b: *std.Build, zantBuild: ZantBuild) void {
    const test_all_oneOp = b.addTest(.{
        .name = "test_all_oneOp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("generated/oneOpModels/test_oneop_models.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    test_all_oneOp.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    test_all_oneOp.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    test_all_oneOp.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //codegen
    test_all_oneOp.root_module.addOptions("testing_options", zantBuild.zantStepOptions.testing_step_option); //<<--OSS!! it is an option!

    test_all_oneOp.linkLibC();

    const run_test_all_oneOp = b.addRunArtifact(test_all_oneOp);

    // ************************************************
    // Setup test oneop
    // It will run
    // - run_oneop_codegen_exe
    // - run_test_all_oneOp

    const step_test_oneOp = b.step("op-codegen-test", "Run generated library tests");
    step_test_oneOp.dependOn(&run_test_all_oneOp.step);
}

inline fn extractor_gen(b: *std.Build, zantBuild: ZantBuild) void {
    const node_extractor_generator = b.addExecutable(.{
        .name = "node_extractor_generator",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/CodeGen/node_extractor_generator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    node_extractor_generator.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    node_extractor_generator.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    node_extractor_generator.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //codegen
    node_extractor_generator.root_module.addOptions("extractor_options", zantBuild.zantStepOptions.extractor_step_option); //<<--OSS!! it is an option!
    node_extractor_generator.linkLibC();

    const run_node_extractor_generator = b.addRunArtifact(node_extractor_generator);
    const step_node_extractor_generator = b.step("extractor-gen", "Codegenerate tests for extracted nodes ");
    step_node_extractor_generator.dependOn(&run_node_extractor_generator.step);
}

inline fn extractor_test(b: *std.Build, zantBuild: ZantBuild) void {
    const test_extractor_path = std.fmt.allocPrint(b.allocator, "generated/{s}/extracted/test_extracted_models.zig", .{zantBuild.zantOptions.codegen_flags.model_name_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating test model path: {}\n", .{err});
        return;
    };

    const test_node_extractor = b.addTest(.{
        .name = "test_node_extractor",
        .root_module = b.createModule(.{
            .root_source_file = b.path(test_extractor_path),
            .target = target,
            .optimize = optimize,
        }),
    });

    test_node_extractor.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    test_node_extractor.root_module.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    test_node_extractor.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //codegen
    test_node_extractor.root_module.addOptions("extractor_options", zantBuild.zantStepOptions.extractor_step_option); //<<--OSS!! it is an option!
    test_node_extractor.linkLibC();

    const run_test_extractor = b.addRunArtifact(test_node_extractor);
    const step_test_extractor = b.step("extractor-test", "Start extracted nodes tests");
    step_test_extractor.dependOn(&run_test_extractor.step);
}

inline fn benchmark_create(b: *std.Build, zantBuild: ZantBuild) void {
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    benchmark.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    benchmark.root_module.addImport("codegen", zantBuild.zantModules.codegen_mod); //codegen
    benchmark.root_module.addOptions("bench_options", zantBuild.zantStepOptions.bench_step_option);
    benchmark.linkLibC();

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);
}

inline fn onnx_parser(b: *std.Build, zantBuild: ZantBuild) void {
    const test_onnx_parser = b.addTest(.{
        .name = "test_generated_lib",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/Onnx/onnx_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    test_onnx_parser.root_module.addImport("zant", zantBuild.zantModules.zant_mod);
    test_onnx_parser.linkLibC();

    const run_test_onnx_parser = b.addRunArtifact(test_onnx_parser);
    const step_test_onnx_parser = b.step("onnx-parser", "Run generated library tests");
    step_test_onnx_parser.dependOn(&run_test_onnx_parser.step);
}

inline fn build_main(b: *std.Build, zantBuild: ZantBuild, static_lib: *std.Build.Step.Compile) void {
    // Path to the generated model options file (moved here)
    const model_options_path = std.fmt.allocPrint(b.allocator, "{s}model_options.zig", .{zantBuild.zantOptions.codegen_flags.generated_path_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating model options path: {}\n", .{err});
        return;
    };

    const main_executable = b.addExecutable(.{
        .name = "main_profiling_target",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    main_executable.linkLibC();
    main_executable.linkLibrary(static_lib);
    const model_opts_mod = b.createModule(.{
        .root_source_file = b.path(model_options_path),
    });
    model_opts_mod.addImport("zant", zantBuild.zantModules.zant_mod);
    model_opts_mod.addImport("codegen", zantBuild.zantModules.codegen_mod);
    model_opts_mod.addImport("IR_zant", zantBuild.zantModules.IR_zant_mod);
    main_executable.root_module.addImport("model_opts", model_opts_mod);
    const install_main_exe_step = b.addInstallArtifact(main_executable, .{}); // Install the executable

    const build_main_step = b.step("build-main", "Build the main executable for profiling");
    build_main_step.dependOn(&install_main_exe_step.step);
}
