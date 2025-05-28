const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const Backend = dvui.backend;
const entypo = dvui.entypo;
comptime {
    std.debug.assert(@hasDecl(Backend, "SDLBackend"));
}

const window_icon_png = @embedFile("zig-favicon.png");
const zant_icon = @embedFile("zant-icon.png");

var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = gpa_instance.allocator();

const vsync = true;
const show_demo = true;
var scale_val: f32 = 1.0;

var show_dialog_outside_frame: bool = false;
var g_backend: ?Backend = null;
var g_win: ?*dvui.Window = null;

/// This example shows how to use the dvui for a normal application:
/// - dvui renders the whole application
/// - render frames only when needed
///
/// Colors
const orange = dvui.Color{ .r = 255, .g = 165, .b = 0, .a = 255 };
const darkorange = dvui.Color{ .r = 172, .g = 99, .b = 22, .a = 255 };
const hoverorange = dvui.Color{ .r = 235, .g = 227, .b = 213, .a = 255 };
const lightorange = dvui.Color{ .r = 255, .g = 246, .b = 229, .a = 255 };

const Page = enum {
    home,
    select_model,
    generating_code,
    deploy_options,
    generating_library,
};
var page: Page = .home;

var filepath: ?[:0]const u8 = null;
var filename: ?[:0]const u8 = null;

const ModelOptions = enum(u8) { default, debug_model, mnist_1, mnist_8, sentiment, wake_word, custom };
var model_options: ModelOptions = @enumFromInt(0);
const model_length = @typeInfo(ModelOptions).@"enum".fields.len;

fn pathToFileName(fp: ?[:0]const u8) [:0]const u8 {
    if (fp == null or fp.?.len == 0) return "";
    const path = fp.?;
    var last_slash: usize = 0;
    for (path, 0..) |c, i| {
        if (c == '/' or c == '\\') {
            last_slash = i + 1;
        }
    }
    return path[last_slash..];
}

fn isOnnx(fp: ?[:0]const u8) bool {
    if (fp) |path| {
        const extension: []const u8 = ".onnx";
        if (path.len >= extension.len) {
            return std.mem.endsWith(u8, path, extension);
        }
    }
    return false;
}

fn getModelString(value: ModelOptions) []const u8 {
    return switch (value) {
        .default => "",
        .debug_model => "Debug Model",
        .mnist_1 => "MNIST-1",
        .mnist_8 => "MNIST-8",
        .sentiment => "Sentiment",
        .wake_word => "Wake Word",
        .custom => {
            if (filename) |name| {
                return name;
            } else {
                return "Not Selected";
            }
        },
    };
}

fn getModelPath(value: ModelOptions) []const u8 {
    return switch (value) {
        .default => "",
        .debug_model => "datasets/models/debug_model/debug_model.onnx",
        .mnist_1 => "datasets/models/mnist-1/mnist-1.onnx",
        .mnist_8 => "datasets/models/mnist-8/mnist-8.onnx",
        .sentiment => "datasets/models/Sentiment/sentiment_analysis_it.onnx",
        .wake_word => "datasets/models/wakeWord/wakeWord.onnx",
        .custom => {
            if (filepath) |fp| {
                return fp;
            } else {
                return "";
            }
        },
    };
}

pub fn pageHome() !void {
    {
        var vbox0 = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.4 });
        defer vbox0.deinit();

        var heading = try dvui.textLayout(@src(), .{}, .{
            .background = false,
            .margin = .{ .h = 20.0 },
        });
        try heading.addText("Z-Ant Simplifies the Deployment\nand Optimization of Neural Networks\non Microprocessors", .{ .font_style = .title });
        heading.deinit();

        if (try (dvui.button(@src(), "Get Started", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) }))) {
            page = .select_model;
        }
    }

    var footer = try dvui.textLayout(@src(), .{}, .{ .background = false, .gravity_x = 0.5, .gravity_y = 0.8 });
    try footer.addText("Z-Ant is an open-source project powered by Zig\nFor help visit our ", .{});
    footer.deinit();
    if (try footer.addTextClick("GitHub", .{ .color_text = .{ .color = orange } })) {
        try dvui.openURL("https://github.com/ZantFoundation/Z-Ant");
    }
}

pub fn pageSelectModel() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        page = .home;
    }

    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.3 });
        defer vbox.deinit();

        try dvui.label(@src(), "Select a Model", .{}, .{ .font_style = .title, .margin = .{ .h = 20.0 }, .gravity_x = 0.5 });

        {
            var vbox1 = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5 });
            defer vbox1.deinit();

            try dvui.label(@src(), "Built in Models", .{}, .{ .font_style = .heading });

            inline for (@typeInfo(ModelOptions).@"enum".fields[1 .. model_length - 1], 0..) |field, i| {
                const enum_value = @as(ModelOptions, @enumFromInt(field.value));
                const display_name = getModelString(enum_value);
                if (try dvui.radio(@src(), model_options == enum_value, display_name, .{ .id_extra = i })) {
                    model_options = enum_value;
                }
            }

            try dvui.label(@src(), "Custom Model", .{}, .{ .font_style = .heading, .margin = .{ .y = 10.0 } });

            if (try dvui.button(@src(), "Open ONNX File", .{}, .{})) {
                if (filepath) |fp| {
                    filename = null;
                    gpa.free(fp);
                    filepath = null;
                }
                filepath = try dvui.dialogNativeFileOpen(gpa, .{ .title = "Pick ONNX File" });
                if (filepath) |fp| {
                    if (isOnnx(fp)) {
                        filename = pathToFileName(fp);
                        model_options = @enumFromInt(model_length - 1);
                    } else {
                        gpa.free(fp);
                        filepath = null;
                        try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "File extension must be .onnx" });
                    }
                }
            }

            const enum_value = @as(ModelOptions, @enumFromInt(model_length - 1));
            const display_name = getModelString(enum_value);
            if (try dvui.radio(@src(), model_options == enum_value, display_name, .{ .id_extra = model_length - 1 })) {
                model_options = enum_value;
            }

            if (try dvui.button(@src(), "Generate Zig Code", .{}, .{ .gravity_x = 0.5, .margin = .{ .y = 20.0 }, .padding = dvui.Rect.all(15) })) {
                if (std.mem.eql(u8, getModelPath(model_options), "")) {
                    try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "You must select a model" });
                } else {
                    var buf: [std.fs.max_path_bytes]u8 = undefined;
                    const exe_dir = try std.fs.selfExeDirPath(&buf);
                    const folder = try std.fs.path.join(std.heap.page_allocator, &[_][]const u8{ exe_dir, "generated/" });
                    const pageallocator = std.heap.page_allocator;
                    var argv2 = [_][]const u8{ "open", folder };
                    var child2 = std.process.Child.init(&argv2, pageallocator);
                    try child2.spawn();
                    page = .generating_code;
                }
            }
        }
    }
}

pub fn pageGeneratingCode() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        page = .select_model;
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.4 });
        defer vbox.deinit();

        try dvui.label(@src(), "Generating Zig Code ...", .{}, .{ .font_style = .heading, .margin = .{ .h = 2.0 } });
        try dvui.label(@src(), "Once completed, the code will be avaialbe in ~/generated", .{}, .{ .margin = .{ .h = 10.0 } });

        if (try dvui.button(@src(), "Continue", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) })) {
            page = .deploy_options;
        }
    }
}

var target_cpu_val: usize = 0;
var target_os_val: usize = 0;

pub fn pageDeployOptions() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        page = .select_model;
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.3 });
        defer vbox.deinit();

        try dvui.label(@src(), "Deploy Options", .{}, .{ .font_style = .title, .margin = .{ .h = 20.0 }, .gravity_x = 0.5 });

        {
            var vbox1 = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5 });
            defer vbox1.deinit();

            const target_cpu = [_][]const u8{ "x86_64", "x86", "arm", "aarch64", "riscv32", "riscv64", "powerpc", "powerpc64", "mips32", "mips64", "wasm32", "wasm64", "sparc", "sparc64" };
            const target_os = [_][]const u8{ "Linux", "Windows", "macOS", "Android", "FreeBSD" };

            try dvui.label(@src(), "Target CPU", .{}, .{ .font_style = .heading, .margin = .{ .h = 5.0 } });
            _ = try dvui.dropdown(@src(), &target_cpu, &target_cpu_val, .{ .min_size_content = .{ .w = 150 }, .margin = .{ .h = 15.0 } });

            try dvui.label(@src(), "Target OS", .{}, .{ .font_style = .heading, .margin = .{ .h = 5.0 } });
            _ = try dvui.dropdown(@src(), &target_os, &target_os_val, .{ .min_size_content = .{ .w = 150 }, .margin = .{ .h = 30.0 } });
        }

        if (try dvui.button(@src(), "Generate Static Library", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) })) {
            page = .generating_library;
        }
    }
}

pub fn pageGeneratingLibrary() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        page = .deploy_options;
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.4 });
        defer vbox.deinit();

        try dvui.label(@src(), "Generating Static Library ...", .{}, .{ .font_style = .heading, .margin = .{ .h = 2.0 } });
        try dvui.label(@src(), "Once completed, the library will be avaialbe in ~/zig-out", .{}, .{ .margin = .{ .h = 10.0 } });

        if (try dvui.button(@src(), "Conclude", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) })) {
            page = .home;
        }
    }
}

pub fn main() !void {
    std.log.info("SDL version: {}", .{Backend.getSDLVersion()});

    dvui.Examples.show_demo_window = show_demo;

    defer if (gpa_instance.deinit() != .ok) @panic("Memory leak on exit!");

    // init SDL backend (creates and owns OS window)
    var backend = try Backend.initWindow(.{
        .allocator = gpa,
        .size = .{ .w = 800.0, .h = 600.0 },
        .min_size = .{ .w = 600.0, .h = 400.0 },
        .vsync = vsync,
        .title = "Z-Ant",
        .icon = window_icon_png, // can also call setIconFromFileContent()
    });
    g_backend = backend;
    defer backend.deinit();

    // init dvui Window (maps onto a single OS window)
    var win = try dvui.Window.init(@src(), gpa, backend.backend(), .{});
    defer win.deinit();

    main_loop: while (true) {

        // beginWait coordinates with waitTime below to run frames only when needed
        const nstime = win.beginWait(backend.hasEvent());

        // marks the beginning of a frame for dvui, can call dvui functions after this
        try win.begin(nstime);

        // send all SDL events to dvui for processing
        const quit = try backend.addAllEvents(&win);
        if (quit) break :main_loop;

        // if dvui widgets might not cover the whole window, then need to clear
        // the previous frame's render
        _ = Backend.c.SDL_SetRenderDrawColor(backend.renderer, 0, 0, 0, 255);
        _ = Backend.c.SDL_RenderClear(backend.renderer);

        // The demos we pass in here show up under "Platform-specific demos"
        try gui_frame();

        // marks end of dvui frame, don't call dvui functions after this
        // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
        const end_micros = try win.end(.{});

        // cursor management
        backend.setCursor(win.cursorRequested());
        backend.textInputRect(win.textInputRequested());

        // render frame to OS
        backend.renderPresent();

        // waitTime and beginWait combine to achieve variable framerates
        const wait_event_micros = win.waitTime(end_micros, null);
        backend.waitEventTimeout(wait_event_micros);

        // Example of how to show a dialog from another thread (outside of win.begin/win.end)
        if (show_dialog_outside_frame) {
            show_dialog_outside_frame = false;
            try dvui.dialog(@src(), .{ .window = &win, .modal = false, .title = "Dialog from Outside", .message = "This is a non modal dialog that was created outside win.begin()/win.end(), usually from another thread." });
        }
    }
}

// both dvui and SDL drawing
fn gui_frame() !void {
    {
        var m = try dvui.menu(@src(), .horizontal, .{ .background = true, .color_fill = .{ .color = lightorange }, .expand = .horizontal });
        defer m.deinit();

        const imgsize = try dvui.imageSize("Z-Ant icon", zant_icon);
        try dvui.image(@src(), "Z-Ant icon", zant_icon, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = imgsize.w * 0.12, .h = imgsize.h * 0.12 },
            .margin = .{ .x = 20, .y = 10, .h = 10.0, .w = 3.0 },
        });
        try dvui.label(@src(), "Z-Ant", .{}, .{ .gravity_y = 0.5, .font_style = .heading });

        if (try dvui.buttonIcon(@src(), "back", entypo.moon, .{}, .{ .background = false, .gravity_y = 0.5, .gravity_x = 1.0, .margin = .{ .w = 20.0 } })) {
            page = .home;
        }
    }

    var scroll = try dvui.scrollArea(@src(), .{}, .{ .expand = .both, .color_fill = .{ .color = dvui.Color.white } });
    defer scroll.deinit();
    switch (page) {
        .home => try pageHome(),
        .select_model => try pageSelectModel(),
        .generating_code => try pageGeneratingCode(),
        .deploy_options => try pageDeployOptions(),
        .generating_library => try pageGeneratingLibrary(),
    }
}
