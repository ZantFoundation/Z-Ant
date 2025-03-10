const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const codegen_options = @import("codegen_options");
const allocator = zant.utils.allocator.allocator;

pub fn writeTestFile(model_name: []const u8, model_path: []const u8, model_onnx: ModelOnnx) !void {

    // TODO: Remove this may be useful in the near future to get some model infos
    _ = model_onnx;

    // Copy test file template into the generated test file

    const test_file_path = try std.fmt.allocPrint(allocator, "{s}test_{s}.zig", .{ model_path, model_name });
    const test_file = try std.fs.cwd().createFile(test_file_path, .{});
    defer test_file.close();
    std.debug.print("\n .......... file created, path:{s}", .{test_file_path});

    var writer: std.fs.File.Writer = test_file.writer();

    const template_file = try std.fs.cwd().openFile("tests/CodeGen/test_model.template.zig", .{});
    defer template_file.close();
    const template_content: []const u8 = try template_file.readToEndAlloc(allocator, 50 * 1024);

    defer allocator.free(template_content);

    try writer.print("{s}", .{template_content});

    // Generate model_options.zig

    const model_options_path = try std.fmt.allocPrint(allocator, "{s}model_options.zig", .{model_path});
    const model_options_file = try std.fs.cwd().createFile(model_options_path, .{});
    defer model_options_file.close();

    writer = model_options_file.writer();

    ////////////

    const input_shape = try utils.parseNumbers(codegen_options.shape);

    _ = try writer.print(
        \\
        \\pub const lib = @import("lib_{s}.zig");
        \\pub const name = "{s}";
        \\pub const input_shape = [{d}]u32{any};
        \\
    , .{
        model_name,
        model_name,
        input_shape.len,
        input_shape,
    });

    ////////////

}
