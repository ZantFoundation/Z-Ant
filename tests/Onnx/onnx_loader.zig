const std = @import("std");
const IR_zant = @import("IR_zant");
const onnx = IR_zant.onnx;
const pkgAllocator = IR_zant.pkg_allocator;
const allocator = pkgAllocator.allocator;

const tests_log = std.log.scoped(.test_utils);

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

test " Onnx loader" {
    tests_log.info("\n     test:  Onnx loader\n", .{});

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var failed_parsed_models: std.ArrayList(ErrorDetail) = .empty;
    defer failed_parsed_models.deinit(allocator);

    var dir = try std.fs.cwd().openDir("datasets/models", .{ .iterate = true });
    defer dir.close();

    // Iterate over directory entries and parse models on the fly
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .directory) continue;

        const model_name = try arena.dupe(u8, entry.name);

        const model_path = try std.mem.concat(allocator, u8, &[_][]const u8{ "datasets/models/", model_name, "/", model_name, ".onnx" });
        defer allocator.free(model_path);

        var model = onnx.parseFromFile(allocator, model_path) catch |err| {
            const errorDetail: ErrorDetail = ErrorDetail{
                .modelName = model_name,
                .errorLoad = err,
            };
            try failed_parsed_models.append(allocator, errorDetail);
            continue;
        };
        defer model.deinit(allocator);
        tests_log.debug("parsed {s}", .{model_name});
    }

    if (failed_parsed_models.items.len != 0) {
        tests_log.info("\n\n FAILED ONNX PARSED MODELS: ", .{});
        for (failed_parsed_models.items) |fm| tests_log.info("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    } else {
        tests_log.info("\n\n ---- SUCCESFULLY PARSED ALL ONNX MODELS ---- \n\n", .{});
    }
}
