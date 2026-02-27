const std = @import("std");
const zant = @import("zant");
const extractor_options = @import("extractor_options");
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const onnx = zant.onnx;
const IR_codeGen = @import("codegen").codegen_v1;

pub fn main() !void {
    std.debug.print("", .{});

    // ----------------------- paths
    const test_dir = try std.fmt.allocPrint(allocator, "generated/{s}/extracted", .{extractor_options.model}); // -> generated/my_model/extracted
    defer allocator.free(test_dir);
    try std.fs.cwd().makePath(test_dir);
    const test_file = try std.fmt.allocPrint(allocator, "{s}/test_extracted_models.zig", .{test_dir}); // -> generated/my_model/extracted/test_extracted_models.zig
    defer allocator.free(test_file);
    const test_minimodels_file = try std.fs.cwd().createFile(test_file, .{});
    defer test_minimodels_file.close();

    const extracted_dir = try std.fmt.allocPrint(allocator, "datasets/models/{s}/extracted_nodes", .{extractor_options.model}); // -> datasets/models/my_model/extracted_nodes
    defer allocator.free(extracted_dir);
    const models_dir = try std.fmt.allocPrint(allocator, "{s}/individual_nodes", .{extracted_dir}); // -> datasets/models/my_model/extracted_nodes/individual_nodes
    defer allocator.free(models_dir);
    const data_tests_dir = try std.fmt.allocPrint(allocator, "{s}/node_data", .{extracted_dir}); // -> datasets/models/my_model/extracted_nodes/node_data
    defer allocator.free(data_tests_dir);

    //check and retrive
    const mini_models = try get_extracted_models(models_dir);

    var test_minimodels_buffer: [1024]u8 = undefined;
    var test_minimodels_writer = test_minimodels_file.writer(&test_minimodels_buffer);
    const writer = &test_minimodels_writer.interface;

    try writer.writeAll("const std = @import(\"std\");\n");
    try writer.writeAll("\n");
    try writer.writeAll("test {");
    try writer.writeAll("\n");

    std.debug.print("\n --- {} models are present \n", .{mini_models.items.len});

    for (mini_models.items) |model_name| {
        std.debug.print("\n\n>>>>>>>>>> {s} ", .{model_name});

        // Construct the model file path: "
        const model_path = try std.fmt.allocPrint(allocator, "{s}/{s}.onnx", .{ models_dir, model_name }); // -> datasets/models/my_model/extracted_nodes/individual_nodes/mini_model.onnx
        defer allocator.free(model_path);
        std.debug.print("model_path : {s}", .{model_path});

        // Load the model.
        var model = try onnx.parseFromFile(allocator, model_path);

        //Printing the model:
        //DEBUG
        //model.print();

        std.debug.print("\n CODEGENERATING {s} ...", .{model_path});
        // CORE PART -------------------------------------------------------
        const codegen_dir = try std.fmt.allocPrint(allocator, "{s}/{s}/", .{ test_dir, model_name }); // -> generated/my_model/extracted/mini_model
        defer allocator.free(codegen_dir);
        try std.fs.cwd().makePath(codegen_dir);
        // ONNX model parsing
        try IR_codeGen.codeGenerateFromOnnx(model_name, codegen_dir, model);

        // Create relative tests
        try IR_codeGen.testWriter.writeSlimTestFile(model_name, codegen_dir); // -> generated/my_model/extracted/mini_model

        // Copy user test file into the generated test file, do not touch, this is not related to model codegen !
        const dataset_test_model_path = try std.fmt.allocPrint(allocator, "{s}/{s}_data.json", .{ data_tests_dir, model_name }); // -> SOURCE
        defer allocator.free(dataset_test_model_path);

        const generated_test_model_path = try std.fmt.allocPrint(allocator, "{s}user_tests.json", .{codegen_dir}); // -> DESTINATION
        defer allocator.free(generated_test_model_path);

        try IR_codeGen.utils.copyFile(dataset_test_model_path, generated_test_model_path);
        std.debug.print("Written user test for {s}", .{model_name});

        // Add relative one op test to global tests file
        try writer.print("\t _ = @import(\"{s}/test_{s}.zig\"); \n", .{ model_name, model_name });

        //try codeGen.globals.setGlobalAttributes(model);
        model.deinit(allocator);
    }

    // Adding last global test line
    try writer.writeAll("} \n\n");
}

// Function to get all .onnx files in a given path
fn get_extracted_models(path: []const u8) !std.ArrayList([]const u8) {
    var model_list: std.ArrayList([]const u8) = .empty;

    var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| {
        std.debug.print("Could not open directory: {s}, error: {}\n", .{ path, err });
        return err;
    };
    defer dir.close();

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".onnx")) {
            const filename = try allocator.dupe(u8, entry.name[0 .. entry.name.len - 5]);
            try model_list.append(allocator, filename);
        }
    }

    if (model_list.items.len == 0) {
        std.debug.print("ERROR: no extracted models found in : {s}\n", .{path});
        return error.NoModelsFound;
    }

    return model_list;
}
