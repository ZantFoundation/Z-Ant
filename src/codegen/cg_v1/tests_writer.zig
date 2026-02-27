const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const codegen_options = @import("codegen_options");
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("IR_zant");

const IR_codegen = IR_zant.IR_codegen;

const IR_utils = IR_zant.utils;

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

pub fn UserTest(comptime T_in: type, comptime T_out: type) type {
    return struct {
        name: []u8,
        type: []u8,
        input: []T_in,
        output: []T_out,
        expected_class: usize,
    };
}

fn writeModelOptionsFile(model_name: []const u8, model_path: []const u8) !void {
    // Generate model_options.zig
    const model_options_path = try std.fmt.allocPrint(allocator, "{s}model_options.zig", .{model_path});
    const model_options_file = try std.fs.cwd().createFile(model_options_path, .{});
    defer model_options_file.close();

    var buffer: [65536]u8 = undefined;
    var model_options_writer = model_options_file.writer(&buffer);
    const writer = &model_options_writer.interface;

    ////////////

    var output_size: i64 = 1;

    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);
    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    for (outputs[0].getShape()) |dim| {
        output_size *= @intCast(dim);
    }

    // Handle case where there are no inputs (e.g., nodes that only have initializers)
    const has_inputs = inputs.len > 0;
    const input_shape_len: usize = if (has_inputs) inputs[0].getShape().len else 0;
    const input_shape: []usize = if (has_inputs) inputs[0].getShape() else &[_]usize{};
    const input_data_type: []const u8 = if (has_inputs) inputs[0].ty.toString() else "f32";

    try writer.print(
        \\
        \\pub const lib = @import("lib_{s}.zig");
        \\pub const name = "{s}";
        \\pub const input_shape = [{d}]u32{any};
        \\pub const output_data_len = {d};
        \\pub const input_data_type = {s};
        \\pub const output_data_type = {s};
        \\pub const has_inputs = {};
        \\pub const user_tests: bool = {};
        \\pub const user_tests_path = "{s}";
        \\pub const have_log: bool = {};
        \\pub const is_dynamic: bool ={};
    , .{
        model_name, //lib
        model_name, //name
        input_shape_len, //input_shape d
        input_shape, //input_shape any
        output_size, //output_data_len
        input_data_type, //input_data_type
        outputs[0].ty.toString(), //output_data_type
        has_inputs, //has_inputs
        codegen_options.user_tests, //user_tests_path
        try std.fmt.allocPrint(allocator, "{s}user_tests.json", .{model_path}), //user_tests_path
        codegen_options.log,
        codegen_options.dynamic,
    });

    ////////////
    try writer.flush();
}

pub fn writeTestFile(model_name: []const u8, model_path: []const u8) !void {

    // Copy test file template into the generated test file
    const test_file_path = try std.fmt.allocPrint(allocator, "{s}test_{s}.zig", .{ model_path, model_name });

    try copyFile("tests/CodeGen/test_model.template.zig", test_file_path);
    std.log.info("\n\nGenerated test file: {s}\n", .{test_file_path});

    // Copy user test file into the generated test file
    if (codegen_options.user_tests) {
        const provided_user_tests_path = try std.fmt.allocPrint(allocator, "datasets/models/{s}/user_tests.json", .{model_name});
        const user_tests_path = try std.fmt.allocPrint(allocator, "{s}user_tests.json", .{model_path});
        try copyFile(provided_user_tests_path, user_tests_path);
    }

    try writeModelOptionsFile(model_name, model_path);
}

pub fn writeSlimTestFile(model_name: []const u8, model_path: []const u8) !void {
    // Copy test file template into the generated test file
    const test_file_path = try std.fmt.allocPrint(allocator, "{s}test_{s}.zig", .{ model_path, model_name });

    try copyFile("tests/CodeGen/test_model.slim.template.zig", test_file_path);
    std.log.info("\n\nGenerated test file: {s}\n", .{test_file_path});

    try writeModelOptionsFile(model_name, model_path);
}

// ----------------- FILE MANAGEMENT -----------------
// Copy file from src to dst
fn copyFile(src_path: []const u8, dst_path: []const u8) !void {
    var src_file = try std.fs.cwd().openFile(src_path, .{
        .mode = .read_only,
    });
    defer src_file.close();

    var dst_file = try std.fs.cwd().createFile(dst_path, .{});
    defer dst_file.close();

    // Use a buffer to copy in chunks
    var buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = try src_file.read(&buf);
        if (bytes_read == 0) break;
        _ = try dst_file.write(buf[0..bytes_read]);
    }
}
