const std = @import("std");

const ZantOptions = @import("zantOptions.zig").ZantOptions;

pub const ZantStepOptions = struct {
    // --- compile steps
    build_step_option: *std.Build.Step.Options, //build_option
    testing_step_option: *std.Build.Step.Options, //testing_option
    codegen_step_option: *std.Build.Step.Options, //codegen_option
    test_step_option: *std.Build.Step.Options, //testing_option ----> deprecated
    extractor_step_option: *std.Build.Step.Options, //extractor_option
    bench_step_option: *std.Build.Step.Options, //bench_option

    pub fn init(b: *std.Build, zantOptions: ZantOptions) !ZantStepOptions {
        return ZantStepOptions{
            .build_step_option = get_build_step_options(b, zantOptions),
            .testing_step_option = get_testing_step_options(b, zantOptions),
            .codegen_step_option = get_codegen_step_options(b, zantOptions),
            .test_step_option = get_test_step_options(b, zantOptions),
            .extractor_step_option = get_extractor_step_options(b, zantOptions),
            .bench_step_option = get_bench_step_options(b, zantOptions),
        };
    }

    fn get_build_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        const build_options: *std.Build.Step.Options = b.addOptions();
        build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
        build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

        build_options.addOption(bool, "use_tensor_pool", zantOptions.codegen_flags.use_tensor_pool); //codegen
        build_options.addOption(bool, "stm32n6_accel", zantOptions.stm32n6_flags.stm32n6_accel); //stm32n6
        build_options.addOption(bool, "stm32n6_force_native", zantOptions.stm32n6_flags.stm32n6_force_native); //stm32n6
        build_options.addOption(bool, "stm32n6_use_cmsis", zantOptions.stm32n6_flags.stm32n6_use_cmsis); //stm32n6
        build_options.addOption(bool, "stm32n6_use_ethos", zantOptions.stm32n6_flags.stm32n6_use_ethos); //stm32n6

        return build_options;
    }

    fn get_testing_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        const testing_flags: *std.Build.Step.Options = b.addOptions();
        testing_flags.addOption([]const u8, "op", zantOptions.testing_flags.op_to_test_option);
        return testing_flags;
    }

    fn get_codegen_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        const codegen_flags: *std.Build.Step.Options = b.addOptions(); // Model name option
        codegen_flags.addOption([]const u8, "model", zantOptions.codegen_flags.model_name_option); //codegen
        codegen_flags.addOption([]const u8, "model_path", zantOptions.codegen_flags.model_path_option); //codegen
        codegen_flags.addOption([]const u8, "generated_path", zantOptions.codegen_flags.generated_path_option); //codegen
        codegen_flags.addOption(bool, "user_tests", zantOptions.codegen_flags.user_tests_option); //codegen
        codegen_flags.addOption(bool, "log", zantOptions.codegen_flags.log_option); //codegen
        codegen_flags.addOption(bool, "do_export", zantOptions.codegen_flags.export_option); //codegen
        codegen_flags.addOption([]const u8, "type", zantOptions.codegen_flags.input_type_option); //codegen
        codegen_flags.addOption([]const u8, "output_type", zantOptions.codegen_flags.output_type_option); //codegen
        codegen_flags.addOption(bool, "comm", zantOptions.codegen_flags.comm_option); //codegen
        codegen_flags.addOption(bool, "dynamic", zantOptions.codegen_flags.dynamic_option); //codegen
        codegen_flags.addOption(bool, "static_planning", zantOptions.codegen_flags.static_planning_option); //codegen
        codegen_flags.addOption(bool, "fuse", zantOptions.codegen_flags.fuse_option); //codegen
        codegen_flags.addOption([]const u8, "version", zantOptions.codegen_flags.codegen_version_option); //codegen
        codegen_flags.addOption(bool, "xip", zantOptions.codegen_flags.xip_enabled); //codegen
        codegen_flags.addOption(bool, "use_tensor_pool", zantOptions.codegen_flags.use_tensor_pool); //codegen
        codegen_flags.addOption(bool, "gen_ino", zantOptions.codegen_flags.gen_ino); //codegen
        codegen_flags.addOption(bool, "gen_h", zantOptions.codegen_flags.gen_h); //codegen
        return codegen_flags;
    }

    fn get_test_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        _ = zantOptions;

        const test_options: *std.Build.Step.Options = b.addOptions();
        test_options.addOption(bool, "heavy", b.option(bool, "heavy", "Run heavy tests") orelse false);
        const test_name = b.option([]const u8, "test_name", "specify a test name to run") orelse "";
        test_options.addOption([]const u8, "test_name", test_name);

        return test_options;
    }

    fn get_extractor_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        const extractor_options: *std.Build.Step.Options = b.addOptions(); // Model name option
        extractor_options.addOption([]const u8, "model", zantOptions.codegen_flags.model_name_option);
        return extractor_options;
    }

    fn get_bench_step_options(b: *std.Build, zantOptions: ZantOptions) *std.Build.Step.Options {
        _ = zantOptions;
        const bench_options: *std.Build.Step.Options = b.addOptions();
        bench_options.addOption(bool, "full", b.option(bool, "full", "Choose whenever run full benchmark or not") orelse false);
        return bench_options;
    }
};
