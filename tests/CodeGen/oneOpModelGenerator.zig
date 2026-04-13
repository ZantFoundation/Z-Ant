//! The aim of this file is that, given a onnx oprator (https://onnx.ai/onnx/operators/),
//! it returns the onnx.ModelProto containing only one node for the operation

const std = @import("std");
const IR_zant = @import("IR_zant");
const testing_options = @import("testing_options");
const pkgAllocator = IR_zant.pkg_allocator;
const allocator = pkgAllocator.allocator;

const onnx = IR_zant.onnx;
const IR_codeGen = @import("codegen").codegen_v1;

// called by "zig build test-codegen-gen" optionals:" -Dlog -Dmodel="name" -D ..." see build.zig in "codegen options"
pub fn main() !void {
    std.debug.print("One ONNX Operator Model Generator", .{});

    // Create folders if not exist
    try std.fs.cwd().makePath("generated/oneOpModels/");

    // There will be a file named test_oneop_models.zig that will contain all the tests for the one operation models.
    // So to test all the models, you can just run the global file
    //
    // Create test_oneop_models.zig
    const test_oneop_file = try std.fs.cwd().createFile("generated/oneOpModels/test_oneop_models.zig", .{});
    defer test_oneop_file.close();

    var test_oneop_file_buffer: [1024]u8 = undefined;
    var test_oneop_writer = test_oneop_file.writer(&test_oneop_file_buffer);
    const writer = &test_oneop_writer.interface;
    try writer.writeAll("const std = @import(\"std\");\n");
    try writer.writeAll("\n");
    try writer.writeAll("test {");
    try writer.writeAll("\n");

    //retrive the operations I want to test
    const operations = try get_operations();

    std.debug.print("\n --- {} operations are present", .{operations.items.len});

    for (operations.items) |op_name| {
        std.debug.print("\n\n >>>>>>>> {s} ", .{op_name});

        // Construct the model file path: "Phython-ONNX/{op}_0.onnx"
        const model_path = try std.fmt.allocPrint(allocator, "datasets/oneOpModels/{s}_0.onnx", .{op_name});
        defer allocator.free(model_path);
        std.debug.print("model_path : {s}", .{model_path});

        // Load the model.
        var model = try onnx.parseFromFile(allocator, model_path);

        //Printing the model:
        //DEBUG
        //model.print();

        std.debug.print("\n CODEGENERATING {s} ...", .{model_path});

        // Create the generated model directory if not present
        const generated_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/", .{op_name});
        defer allocator.free(generated_path);
        try std.fs.cwd().makePath(generated_path);

        // CORE PART -------------------------------------------------------
        // ONNX model parsing
        try IR_codeGen.codeGenerateFromOnnx(op_name, generated_path, model);

        // Create relative tests
        try IR_codeGen.testWriter.writeSlimTestFile(op_name, generated_path);

        // Copy user test file into the generated test file, do not touch, this is not related to model codegen !
        const dataset_test_model_path = try std.fmt.allocPrint(allocator, "datasets/oneOpModels/{s}_0_user_tests.json", .{op_name});
        defer allocator.free(dataset_test_model_path);

        const generated_test_model_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/user_tests.json", .{op_name});
        defer allocator.free(generated_test_model_path);

        try IR_codeGen.utils.copyFile(dataset_test_model_path, generated_test_model_path);
        std.debug.print("Written user test for {s}", .{op_name});

        // Add relative one op test to global tests file
        try writer.print("\t _ = @import(\"{s}/test_{s}.zig\"); \n", .{ op_name, op_name });

        //try codeGen.globals.setGlobalAttributes(model);
        model.deinit(allocator);
    }

    // Adding last global test line
    try writer.writeAll("} \n\n");

    try writer.flush();
}

fn get_operations() !std.ArrayList([]const u8) {
    var op_list: std.ArrayList([]const u8) = .empty;

    if (!std.mem.eql(u8, testing_options.op, "all")) {
        try op_list.append(allocator, testing_options.op);
    } else {
        //collecting available operations from tests/CodeGen/Python-ONNX/available_operations.txt
        std.debug.print("\n     opening available_operations...", .{});
        const op_file = try std.fs.cwd().openFile("tests/CodeGen/Python-ONNX/available_operations.txt", .{});
        defer op_file.close();
        std.debug.print(" done", .{});

        const file_size = try op_file.getEndPos();
        const buffer = try allocator.alloc(u8, @intCast(file_size));

        const bytes_read = try op_file.readAll(buffer);
        if (bytes_read != file_size) {
            return error.UnexpectedEOF;
        }

        // Read the available operations from file.
        const ops_data = buffer[0..file_size];
        defer allocator.free(ops_data);

        // Split file into lines.
        var lines_iter = std.mem.splitAny(u8, ops_data, "\n");

        while (true) {
            // Get the next line from the iterator.
            const maybe_line = lines_iter.next();

            if (maybe_line) |_| std.debug.print("\n", .{}) else {
                std.debug.print("NO MORE OPERATIONS -----> break\n", .{});
                break;
            }

            const raw_line = maybe_line.?;

            // if it is a comment or an empty line ignore it
            if (raw_line[0] == '#' or raw_line[0] == '\n') continue;

            // Trim whitespace from the line.
            const trimmed_line = std.mem.trim(u8, raw_line, " \t\r\n");
            if (trimmed_line.len > 0) {
                std.debug.print(" ############ Loading Operation: {s} ############\n", .{trimmed_line});
                const copy = try allocator.alloc(u8, trimmed_line.len);
                @memcpy(copy, trimmed_line);
                try op_list.append(allocator, copy);
            }
        }
    }

    return op_list;
}
