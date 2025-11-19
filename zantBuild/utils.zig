const std = @import("std");

const Stm32n6_flags = @import("stm32n6_flags.zig").Stm32n6_flags;
const Cmsis_flags = @import("cmsis_flags.zig").Cmsis_flags;

pub fn configureCmsisSupport(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    cmsis_flags: Cmsis_flags,
) void {
    _ = cmsis_flags;

    //const cmsis_path: ?[]const u8 = cmsis_flags.cmsis_path;
    const cmsis_path = null;

    if (cmsis_path) |path| {
        step.addIncludePath(.{ .cwd_relative = path });
        step.addIncludePath(.{ .cwd_relative = std.fmt.allocPrint(b.allocator, "{s}/Core/Include", .{path}) catch unreachable });
    } else {
        if (std.fs.cwd().access("third_party/CMSIS-NN", .{})) |_| {
            step.addIncludePath(b.path("third_party/CMSIS-NN"));
            step.addIncludePath(b.path("third_party/CMSIS-NN/Include"));
        } else |err| {
            if (err != error.FileNotFound) @panic("unexpected error probing CMSIS-NN path");
        }
        if (std.fs.cwd().access("third_party/CMSIS_5/CMSIS/Core/Include", .{})) |_| {
            step.addIncludePath(b.path("third_party/CMSIS_5/CMSIS/Core/Include"));
        } else |err| {
            if (err != error.FileNotFound) @panic("unexpected error probing CMSIS Core path");
        }
    }

    if (std.fs.cwd().access("third_party/CMSIS-DSP/Include", .{})) |_| {
        step.addIncludePath(b.path("third_party/CMSIS-DSP/Include"));
    } else |err| {
        if (err != error.FileNotFound) @panic("unexpected error probing CMSIS-DSP path");
    }

    if (std.fs.cwd().access("third_party/CMSIS-DSP/PrivateInclude", .{})) |_| {
        step.addIncludePath(b.path("third_party/CMSIS-DSP/PrivateInclude"));
    } else |err| {
        if (err != error.FileNotFound) @panic("unexpected error probing CMSIS-DSP PrivateInclude path");
    }

    // Add ARM newlib headers so <string.h>, <math.h>, etc. are found when targeting arm-none-eabi
    if (std.fs.cwd().access("/usr/lib/arm-none-eabi/include", .{})) {
        step.addIncludePath(.{ .cwd_relative = "/usr/lib/arm-none-eabi/include" });
    } else |_| {
        // Fallback common location (ignore errors)
        if (std.fs.cwd().access("/usr/arm-none-eabi/include", .{})) {
            step.addIncludePath(.{ .cwd_relative = "/usr/arm-none-eabi/include" });
        } else |_| {}
    }

    // Add CMSIS-NN source files
    if (std.fs.cwd().access("third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c", .{})) |_| {
        step.addCSourceFile(.{
            .file = b.path("third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c"),
            //.flags = c_flags,
        });
    } else |err| {
        if (err != error.FileNotFound) @panic("unexpected error probing arm_convolve_s8.c");
    }

    if (std.fs.cwd().access("third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c", .{})) |_| {
        step.addCSourceFile(.{
            .file = b.path("third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c"),
            //.flags = c_flags,
        });
    } else |err| {
        if (err != error.FileNotFound) @panic("unexpected error probing arm_convolve_get_buffer_sizes_s8.c");
    }

    // Add additional CMSIS-NN source files that are commonly needed
    const cmsis_nn_sources = [_][]const u8{
        // Convolution functions files
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c",
        // Support functions files
        "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
        "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
        "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
        "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.c",
        "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    };

    for (cmsis_nn_sources) |source_path| {
        if (std.fs.cwd().access(source_path, .{})) |_| {
            step.addCSourceFile(.{
                .file = b.path(source_path),
                //.flags = c_flags,
            });
        } else |err| {
            if (err != error.FileNotFound) @panic("unexpected error probing CMSIS-NN source");
        }
    }

    // Add CMSIS-DSP source files
    const cmsis_dsp_sources = [_][]const u8{
        "third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c",
    };

    for (cmsis_dsp_sources) |source_path| {
        if (std.fs.cwd().access(source_path, .{})) |_| {
            step.addCSourceFile(.{
                .file = b.path(source_path),
                //.flags = c_flags,
            });
        } else |err| {
            if (err != error.FileNotFound) @panic("unexpected error probing CMSIS-DSP source");
        }
    }
}

pub fn configureStm32n6Support(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    stm32n6_flags: Stm32n6_flags,
) void {
    const use_ethos: bool = stm32n6_flags.stm32n6_use_ethos;
    const ethos_path: ?[]const u8 = stm32n6_flags.stm32n6_ethos_path;
    const force_native: bool = stm32n6_flags.stm32n6_force_native;

    step.addIncludePath(b.path("src/Core/Tensor/Accelerators/stm32n6"));

    var c_flag_buf: [3][]const u8 = undefined;
    var flag_buf = std.ArrayListUnmanaged([]const u8).initBuffer(&c_flag_buf);
    if (force_native) flag_buf.append(b.allocator, "-DZANT_STM32N6_FORCE_NATIVE=1") catch unreachable;
    if (use_ethos) flag_buf.append(b.allocator, "-DZANT_HAS_ETHOS_U=1") catch unreachable;
    const c_flags = flag_buf.items;

    // step.addCSourceFile(.{
    //     .file = b.path("src/Core/Tensor/Accelerators/stm32n6/conv_f32.c"),
    //     .flags = c_flags,
    // });
    step.addCSourceFile(.{
        .file = b.path("src/Core/Tensor/Accelerators/stm32n6/ethos_stub.c"),
        .flags = c_flags,
    });

    if (use_ethos) {
        if (ethos_path) |path| {
            step.addIncludePath(.{ .cwd_relative = path });
        }
    }
}
