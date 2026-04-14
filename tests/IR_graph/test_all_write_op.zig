const testing = std.testing;
const std = @import("std");
const IR_zant = @import("IR_zant");
const onnx = IR_zant.onnx;
const allocator = IR_zant.pkg_allocator.allocator;
const fs = std.fs;

const Tensor = IR_zant.core.tensor.Tensor;

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = .empty;
var failed_parsed_models: std.ArrayList(ErrorDetail) = .empty;
var failed_write_op_models: std.ArrayList(ErrorDetail) = .empty;

test "Test write_op on all oneOp models" {
    std.debug.print("\n     test: Test write_op on all oneOp models\n", .{});

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

    // Create folders if not exist
    try std.fs.cwd().makePath("generated/oneOpModels_graphZant/");

    // Create test_oneop_models.zig
    //const test_oneop_file = try std.fs.cwd().createFile("generated/oneOpModels_graphZant/test_oneop_models.zig", .{});

    while (true) {
        // Get the next line from the iterator.
        const maybe_line = lines_iter.next();

        if (maybe_line) |_| std.debug.print("\n\n", .{}) else {
            std.debug.print(" \n\n no more models -----> break\n", .{});
            break;
        }

        const raw_line = maybe_line.?;

        // if it is a comment or an empty line ignore it
        if (raw_line[0] == '#' or raw_line[0] == '\n') continue;

        // Trim whitespace from the line.
        const trimmed_line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (trimmed_line.len > 0) {
            std.debug.print(" ############ Operation: {s} ############ \n", .{trimmed_line});
        }

        const model_name = trimmed_line;

        // Construct the model file path: "Phython-ONNX/{op}_0.onnx"
        const model_path = try std.fmt.allocPrint(allocator, "datasets/oneOpModels/{s}_0.onnx", .{model_name});
        defer allocator.free(model_path);
        std.debug.print("model_path : {s}", .{model_path});

        // Create the generated model directory if not present
        const generated_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels_graphZant/{s}/", .{model_name});
        defer allocator.free(generated_path);
        try std.fs.cwd().makePath(generated_path);

        // Create a temporary file for write_op output
        const temp_file_path = try std.fmt.allocPrint(allocator, "{s}{s}_graphZant.zig", .{ generated_path, model_name });
        defer allocator.free(temp_file_path);
        var file = try std.fs.cwd().createFile(temp_file_path, .{});

        var file_writer_buffer: [4096]u8 = undefined;
        var file_writer = file.writer(&file_writer_buffer);
        const writer = &file_writer.interface;
        try writer.writeAll("// Generated test file \n");

        // Parse the model
        var model = onnx.parseFromFile(allocator, model_path) catch |err| {
            std.debug.print("\nError parsing model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(allocator, .{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer model.deinit(allocator);

        try writer.print("\n\n// ---------- onnx Model: {s} ----------\n", .{model.graph.?.name.?});
        std.debug.print("\n\n - onnx Model: {s} --------------------------------n", .{model.graph.?.name.?});

        // Create GraphZant
        var graphZant = IR_zant.init(&model) catch |err| {
            std.debug.print("\nError creating graph for model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(allocator, .{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer graphZant.deinit();

        // Linearize the graph
        const linearizedGraph = graphZant.linearize(allocator) catch |err| {
            std.debug.print("\nError linearizing graph for model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(allocator, .{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer linearizedGraph.deinit();

        std.debug.print("\n  Linearized Graph for {s} has {d} nodes\n", .{ model_name, linearizedGraph.items.len });

        // Test write_op on each node
        for (linearizedGraph.items, 0..) |node, i| {
            const node_name = node.name orelse "<unnamed>";
            std.debug.print("\n   - Testing node {d}: {s} (type: {s})", .{ i, node_name, node.op_type });

            try writer.print("\n// Node {d}: {s} (type: {s})\n", .{ i, node_name, node.op_type });

            // Try to call write_op on this node
            node.write_op(writer) catch |err| {
                std.debug.print("\n     ERROR: Failed to write_op for node {s} in model {s}: {any}\n", .{ node_name, model_name, err });
                try failed_write_op_models.append(allocator, .{ .modelName = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ model_name, node_name }), .errorLoad = err });
                try writer.print("// ERROR: write_op failed with error: {any}\n", .{err});
                continue;
            };

            try writer.print("// write_op completed successfully\n", .{});
        }
    }

    // ----------- PRINTING THE RESULTS-----------
    if (failed_parsed_models.items.len != 0) {
        std.debug.print("\n\n FAILED ONNX PARSED MODELS: ", .{});
        for (failed_parsed_models.items) |fm|
            std.debug.print("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    }

    if (failed_write_op_models.items.len != 0) {
        std.debug.print("\n\n FAILED WRITE_OP OPERATIONS: ", .{});
        for (failed_write_op_models.items) |fm|
            std.debug.print("\n model/node:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    }

    if (failed_parsed_models.items.len == 0 and failed_write_op_models.items.len == 0) {
        std.debug.print("\n\n ---- SUCCESSFULLY TESTED WRITE_OP ON ALL ONE-OP MODELS ---- \n\n", .{});
    }

    // Clean up
    for (models.items) |name| {
        allocator.free(name);
    }
    models.deinit(allocator);

    for (failed_parsed_models.items) |item| {
        allocator.free(item.modelName);
    }
    failed_parsed_models.deinit(allocator);

    for (failed_write_op_models.items) |item| {
        allocator.free(item.modelName);
    }
    failed_write_op_models.deinit(allocator);
}
