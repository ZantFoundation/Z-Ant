const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .sdl3 = true });

    const examples = [_][]const u8{
        "sdl-standalone",
    };

    inline for (examples) |ex| {
        const exe = b.addExecutable(.{
            .name = ex,
            .root_source_file = b.path(ex ++ ".zig"),
            .target = target,
            .optimize = optimize,
        });

        // Can either link the backend ourselves:
        // const dvui_mod = dvui_dep.module("dvui");
        // const sdl = dvui_dep.module("sdl");
        // @import("dvui").linkBackend(dvui_mod, sdl);
        // exe.root_module.addImport("dvui", dvui_mod);

        // Or use a prelinked one:
        exe.root_module.addImport("dvui", dvui_dep.module("dvui_sdl"));

        const compile_step = b.step("compile-" ++ ex, "Compile " ++ ex);
        compile_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        b.getInstallStep().dependOn(compile_step);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(compile_step);

        const run_step = b.step(ex, "Run " ++ ex);
        run_step.dependOn(&run_cmd.step);
    }
}
