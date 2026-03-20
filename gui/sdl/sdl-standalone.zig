const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const Backend = dvui.backend;
const entypo = dvui.entypo;
const Color = dvui.Color;
//const zant = @import("zant");

comptime {
    std.debug.assert(@hasDecl(Backend, "SDLBackend"));
}

const window_icon_png = @embedFile("zant-favicon.png");
const zant_icon = @embedFile("zant-icon.png");

var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
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
// Colors
const orange50 = Color{ .r = 255, .g = 252, .b = 234, .a = 255 };
const orange100 = Color{ .r = 255, .g = 245, .b = 197, .a = 255 };
const orange200 = Color{ .r = 255, .g = 235, .b = 133, .a = 255 };
const orange300 = Color{ .r = 255, .g = 219, .b = 70, .a = 255 };
const orange400 = Color{ .r = 255, .g = 200, .b = 27, .a = 255 };
const orange500 = Color{ .r = 255, .g = 166, .b = 2, .a = 255 };
const orange600 = Color{ .r = 226, .g = 125, .b = 0, .a = 255 };
const orange700 = Color{ .r = 187, .g = 86, .b = 2, .a = 255 };
const orange800 = Color{ .r = 152, .g = 66, .b = 8, .a = 255 };
const orange900 = Color{ .r = 124, .g = 54, .b = 11, .a = 255 };
const orange950 = Color{ .r = 72, .g = 26, .b = 0, .a = 255 };
const transparent = Color{ .r = 0, .g = 0, .b = 0, .a = 0 };
const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
const black = Color{ .r = 24, .g = 24, .b = 27, .a = 255 };
const border_light = Color{ .r = 212, .g = 212, .b = 216, .a = 255 };
const border_dark = Color{ .r = 39, .g = 39, .b = 42, .a = 255 };
const grey_light = Color{ .r = 249, .g = 249, .b = 249, .a = 255 };
const grey_dark = Color{ .r = 32, .g = 32, .b = 35, .a = 255 };
const button_normal_light = Color{ .r = 240, .g = 240, .b = 240, .a = 255 };
const button_hover_light = Color{ .r = 225, .g = 225, .b = 225, .a = 255 };
const button_pressed_light = Color{ .r = 200, .g = 200, .b = 200, .a = 255 };
const button_normal_dark = Color{ .r = 50, .g = 50, .b = 50, .a = 255 };
const button_hover_dark = Color{ .r = 70, .g = 70, .b = 70, .a = 255 };
const button_pressed_dark = Color{ .r = 100, .g = 100, .b = 100, .a = 255 };
var background_color = white;
var menubar_color = orange50;

// Theme

var first = true;
var darkmode = false;

fn applyTheme() void {
    const theme = dvui.themeGet();
    if (!darkmode) {
        theme.dark = false;
        theme.color_accent = orange500;
        //theme.color_err = red;
        theme.color_text = black;
        theme.color_text_press = black;
        theme.color_fill = white;
        theme.color_fill_window = grey_light;
        theme.color_fill_control = button_normal_light;
        theme.color_fill_hover = button_hover_light;
        theme.color_fill_press = button_pressed_light;
        theme.color_border = border_light;
        background_color = white;
        menubar_color = orange50;
    } else {
        theme.dark = true;
        theme.color_accent = orange500;
        //theme.color_err = red;
        theme.color_text = white;
        theme.color_text_press = white;
        theme.color_fill = black;
        theme.color_fill_window = grey_dark;
        theme.color_fill_control = button_normal_dark;
        theme.color_fill_hover = button_hover_dark;
        theme.color_fill_press = button_pressed_dark;
        theme.color_border = border_dark;
        background_color = black;
        menubar_color = orange950;
    }
}

// Global variables

const Page = enum {
    home,
    select_model,
    generating_code,
    deploy_options,
    generating_library,
};
var page: Page = .home;

var filepath: ?[:0]const u8 = null;
var filename: ?[]const u8 = null;

const ModelOptions = enum(u8) { default, debug_model, mnist_1, mnist_8, sentiment, wake_word, custom };
var model_options: ModelOptions = @enumFromInt(0);
const model_length = @typeInfo(ModelOptions).@"enum".fields.len;

var target_cpu_val: usize = 0;
var target_os_val: usize = 0;

// Helper functions

fn pathToFileName(fp: ?[:0]const u8) []const u8 {
    if (fp == null or fp.?.len == 0) return "";
    const path = fp.?;
    var last_slash: usize = 0;
    for (path, 0..) |c, i| {
        if (c == '/' or c == '\\') {
            last_slash = i + 1;
        }
    }
    var last_dot = path.len;
    for (path[last_slash..], 0..) |c, i| {
        if (c == '.') {
            last_dot = last_slash + i;
        }
    }
    return path[last_slash..last_dot];
}

fn isOnnx(fp: ?[:0]const u8) bool {
    if (fp) |path| {
        const forbidden_chars = [_]u8{ '|', '&', ';', '$', '`', '>', '<' };
        for (forbidden_chars) |c| {
            if (std.mem.indexOfScalar(u8, path, c) != null) {
                return false;
            }
        }
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

fn getModelName(value: ModelOptions) []const u8 {
    return switch (value) {
        .default => "",
        .debug_model => "debug_model",
        .mnist_1 => "mnist-1",
        .mnist_8 => "mnist-8",
        .sentiment => "sentiment_it",
        .wake_word => "wakeWord",
        .custom => {
            if (filename) |name| {
                return name;
            } else {
                return "";
            }
        },
    };
}

// Pages

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

        if (try (dvui.button(@src(), "Get Started", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15), .color_fill = .{ .color = orange500 }, .color_fill_hover = .{ .color = orange600 }, .color_fill_press = .{ .color = orange700 }, .color_text = .{ .color = orange950 } }))) {
            page = .select_model;
        }
    }

    var footer = try dvui.textLayout(@src(), .{}, .{
        .background = false,
        .gravity_x = 0.5,
        .gravity_y = 0.8,
    });
    try footer.addText("Z-Ant is an open-source project powered by Zig\nFor help visit our ", .{});
    footer.deinit();
    if (try footer.addTextClick("GitHub", .{ .color_text = .{ .color = orange500 } })) {
        try dvui.openURL("https://github.com/ZantFoundation/Z-Ant");
    }
}

var generating = std.atomic.Value(bool).init(false);

fn runCodeGen() !void {
    if (model_options == .custom) {
        if (filepath) |fp| {
            const model_dir = try std.fmt.allocPrint(gpa, "datasets/models/{s}", .{getModelName(model_options)});
            defer gpa.free(model_dir);
            try std.fs.cwd().makePath(model_dir);
            try std.fs.cwd().copyFile(fp, std.fs.cwd(), try std.fmt.allocPrint(gpa, "{s}/{s}.onnx", .{ model_dir, pathToFileName(fp) }), .{});
        }
    }

    const model_name = getModelName(model_options);
    const model_flag = try std.fmt.allocPrint(gpa, "-Dmodel={s}", .{model_name});
    defer gpa.free(model_flag);
    var argv = [_][]const u8{ "zig", "build", "lib", model_flag };
    var child = std.process.Child.init(&argv, gpa);

    try child.spawn();

    const exit_status = try child.wait();

    std.log.info("\nExit Status: {}\n", .{exit_status});
    if (exit_status.Exited != 0) {
        page = .select_model;
    }
    generating.store(false, .release);
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

            if (try dvui.button(@src(), "Generate Zig Code", .{}, .{ .gravity_x = 0.5, .margin = .{ .y = 20.0 }, .padding = dvui.Rect.all(15), .color_fill = .{ .color = orange500 }, .color_fill_hover = .{ .color = orange600 }, .color_fill_press = .{ .color = orange700 }, .color_text = .{ .color = orange950 } })) {
                if (std.mem.eql(u8, getModelName(model_options), "")) {
                    try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "You must select a model" });
                } else {
                    generating.store(true, .monotonic);
                    _ = try std.Thread.spawn(.{}, runCodeGen, .{});
                    page = .generating_code;
                }
            }
        }
    }
}

pub fn pageGeneratingCode() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        if (generating.load(.acquire)) {
            try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "Wait for code generation to complete" });
        } else {
            page = .select_model;
        }
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.4 });
        defer vbox.deinit();
        if (generating.load(.acquire)) {
            try dvui.label(@src(), "Generating Zig Code ...", .{}, .{
                .font_style = .heading,
                .margin = .{ .h = 2.0 },
            });
        } else {
            try dvui.label(@src(), "Generated Zig Code", .{}, .{
                .font_style = .heading,
                .margin = .{ .h = 2.0 },
            });
        }
        try dvui.label(@src(), "Once completed, the code will be avaialbe in ~/generated", .{}, .{ .margin = .{ .h = 10.0 } });

        {
            var hbox = try dvui.box(@src(), .horizontal, .{ .gravity_x = 0.5 });
            defer hbox.deinit();

            if (try dvui.button(@src(), "Open Folder", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) })) {
                var argv = [_][]const u8{ "open", "generated" };
                var child = std.process.Child.init(&argv, gpa);
                try child.spawn();
            }
            if (try dvui.button(@src(), "Continue", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15), .color_fill = .{ .color = orange500 }, .color_fill_hover = .{ .color = orange600 }, .color_fill_press = .{ .color = orange700 }, .color_text = .{ .color = orange950 } })) {
                if (generating.load(.acquire)) {
                    try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "Wait for code generation to complete" });
                } else {
                    page = .deploy_options;
                }
            }
        }
    }
}

var target_arch_str: ?[:0]const u8 = null;
var target_cpu_str: ?[:0]const u8 = null;

pub fn runLibGen() !void {
    const model_name = getModelName(model_options);
    const model_flag = try std.fmt.allocPrint(gpa, "-Dmodel={s}", .{model_name});
    defer gpa.free(model_flag);

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(gpa);
    try args.appendSlice(gpa, &[_][]const u8{ "zig", "build", "lib", model_flag });

    var arch_flag: ?[]const u8 = null;
    if (target_arch_str) |str| {
        if (!std.mem.eql(u8, str, "")) {
            arch_flag = try std.fmt.allocPrint(gpa, "-Dtarget={s}", .{str});
            try args.append(gpa, arch_flag.?);
        }
    }
    defer if (arch_flag) |flag| gpa.free(flag);

    var cpu_flag: ?[]const u8 = null;
    if (target_cpu_str) |str| {
        if (!std.mem.eql(u8, str, "")) {
            cpu_flag = try std.fmt.allocPrint(gpa, "-Dcpu={s}", .{str});
            try args.append(gpa, cpu_flag.?);
        }
    }
    defer if (cpu_flag) |flag| gpa.free(flag);

    var child = std.process.Child.init(args.items, gpa);

    try child.spawn();

    const exit_status = try child.wait();

    std.log.info("\nExit Status: {}\n", .{exit_status});
    if (exit_status.Exited != 0) {
        page = .deploy_options;
    }
    generating.store(false, .release);

    if (target_arch_str) |str| {
        gpa.free(str);
    }
    if (target_cpu_str) |str| {
        gpa.free(str);
    }
}

pub fn pageDeployOptions() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        page = .select_model;
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.3 });
        defer vbox.deinit();

        try dvui.label(@src(), "Deploy Options", .{}, .{ .font_style = .title, .margin = .{ .h = 20.0 }, .gravity_x = 0.5 });
        var footer = try dvui.textLayout(@src(), .{}, .{
            .background = false,
            .gravity_x = 0.5,
        });
        try footer.addText("Leave blank to deploy to host machine\nList of ", .{});
        footer.deinit();
        if (try footer.addTextClick("accepted devices", .{ .color_text = .{ .color = orange500 } })) {
            try dvui.openURL("https://github.com/ZantFoundation/Z-Ant");
        }

        {
            var vbox1 = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5 });
            defer vbox1.deinit();

            try dvui.label(@src(), "Target Architecture", .{}, .{ .font_style = .heading });
            var target_arch = try dvui.textEntry(@src(), .{}, .{ .gravity_x = 0.5, .margin = .{ .h = 10.0 } });
            target_arch.deinit();
            try dvui.label(@src(), "Specific CPU", .{}, .{ .font_style = .heading });
            var target_cpu = try dvui.textEntry(@src(), .{}, .{ .gravity_x = 0.5, .margin = .{ .h = 20.0 } });
            target_cpu.deinit();

            if (try dvui.button(@src(), "Generate Static Library", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15), .color_fill = .{ .color = orange500 }, .color_fill_hover = .{ .color = orange600 }, .color_fill_press = .{ .color = orange700 }, .color_text = .{ .color = orange950 } })) {
                generating.store(true, .monotonic);
                target_arch_str = try gpa.dupeZ(u8, target_arch.getText());
                target_cpu_str = try gpa.dupeZ(u8, target_cpu.getText());
                _ = try std.Thread.spawn(.{}, runLibGen, .{});
                page = .generating_library;
            }
        }
    }
}

pub fn pageGeneratingLibrary() !void {
    if (try dvui.buttonIcon(@src(), "back", entypo.chevron_left, .{}, .{ .margin = dvui.Rect.all(15) })) {
        if (generating.load(.acquire)) {
            try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "Wait for library generation to complete" });
        } else {
            page = .deploy_options;
        }
    }
    {
        var vbox = try dvui.box(@src(), .vertical, .{ .gravity_x = 0.5, .gravity_y = 0.4 });
        defer vbox.deinit();
        if (generating.load(.acquire)) {
            try dvui.label(@src(), "Generating Static Library ...", .{}, .{
                .font_style = .heading,
                .margin = .{ .h = 2.0 },
            });
        } else {
            try dvui.label(@src(), "Generated Static Library", .{}, .{
                .font_style = .heading,
                .margin = .{ .h = 2.0 },
            });
        }
        try dvui.label(@src(), "Once completed, the code will be avaialbe in ~/zig-out", .{}, .{ .margin = .{ .h = 10.0 } });

        {
            var hbox = try dvui.box(@src(), .horizontal, .{ .gravity_x = 0.5 });
            defer hbox.deinit();

            if (try dvui.button(@src(), "Open Folder", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15) })) {
                var argv = [_][]const u8{ "open", "zig-out" };
                var child = std.process.Child.init(&argv, gpa);
                try child.spawn();
            }

            if (try dvui.button(@src(), "Conclude", .{}, .{ .gravity_x = 0.5, .padding = dvui.Rect.all(15), .color_fill = .{ .color = orange500 }, .color_fill_hover = .{ .color = orange600 }, .color_fill_press = .{ .color = orange700 }, .color_text = .{ .color = orange950 } })) {
                if (generating.load(.acquire)) {
                    try dvui.dialog(@src(), .{ .modal = true, .title = "Error", .message = "Wait for library generation to complete" });
                } else {
                    page = .home;
                }
            }
        }
    }
}

pub fn main() !void {
    if (@import("builtin").os.tag == .windows) { // optional
        // on windows graphical apps have no console, so output goes to nowhere - attach it manually. related: https://github.com/ziglang/zig/issues/4196
        _ = winapi.AttachConsole(0xFFFFFFFF);
    }
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
    if (first) {
        applyTheme();
        first = false;
    }
    {
        var m = try dvui.menu(@src(), .horizontal, .{ .background = true, .color_fill = .{ .color = menubar_color }, .expand = .horizontal });
        defer m.deinit();

        const imgsize = try dvui.imageSize("Z-Ant icon", zant_icon);
        try dvui.image(@src(), "Z-Ant icon", zant_icon, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = imgsize.w * 0.12, .h = imgsize.h * 0.12 },
            .margin = .{ .x = 20, .y = 10, .h = 10.0, .w = 3.0 },
        });
        try dvui.label(@src(), "Z-Ant", .{}, .{ .gravity_y = 0.5, .font_style = .heading });

        if (try dvui.buttonIcon(@src(), "back", entypo.adjust, .{}, .{ .background = false, .gravity_y = 0.5, .gravity_x = 1.0, .margin = .{ .w = 20.0 }, .color_accent = .{ .color = transparent } })) {
            darkmode = !darkmode;
            applyTheme();
        }
    }

    var scroll = try dvui.scrollArea(@src(), .{}, .{ .expand = .both, .color_fill = .{ .color = background_color } });
    defer scroll.deinit();
    switch (page) {
        .home => try pageHome(),
        .select_model => try pageSelectModel(),
        .generating_code => try pageGeneratingCode(),
        .deploy_options => try pageDeployOptions(),
        .generating_library => try pageGeneratingLibrary(),
    }
}

// Optional: windows os only
const winapi = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn AttachConsole(dwProcessId: std.os.windows.DWORD) std.os.windows.BOOL;
} else struct {};
