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

    // Outer arena holds string data (names, error records) that must survive
    // across iterations. Parse-time allocations go into a per-iteration arena
    // so they're released regardless of whether ModelProto.parse errored part
    // way through (partial-parse cleanup on error is not handled by the parser).
    var outer_arena = std.heap.ArenaAllocator.init(allocator);
    defer outer_arena.deinit();
    const outer = outer_arena.allocator();

    var failed_parsed_models: std.ArrayList(ErrorDetail) = .empty;
    defer failed_parsed_models.deinit(outer);

    var dir = try std.fs.cwd().openDir("datasets/models", .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .directory) continue;

        const model_name = try outer.dupe(u8, entry.name);

        var parse_arena = std.heap.ArenaAllocator.init(allocator);
        defer parse_arena.deinit();
        const parse_alloc = parse_arena.allocator();

        const model_path = try std.mem.concat(parse_alloc, u8, &[_][]const u8{ "datasets/models/", model_name, "/", model_name, ".onnx" });

        _ = onnx.parseFromFile(parse_alloc, model_path) catch |err| {
            try failed_parsed_models.append(outer, .{ .modelName = model_name, .errorLoad = err });
            continue;
        };
        tests_log.debug("parsed {s}", .{model_name});
    }

    if (failed_parsed_models.items.len != 0) {
        tests_log.info("\n\n FAILED ONNX PARSED MODELS: ", .{});
        for (failed_parsed_models.items) |fm| tests_log.info("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    } else {
        tests_log.info("\n\n ---- SUCCESFULLY PARSED ALL ONNX MODELS ---- \n\n", .{});
    }
}
