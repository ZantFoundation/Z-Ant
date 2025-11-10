const std = @import("std");

/// Helper that replaces text and frees the previous buffer
fn replaceAndFree(
    alloc: std.mem.Allocator,
    buf: []u8,
    needle: []const u8,
    repl: []const u8,
) ![]u8 {
    const out = try std.mem.replaceOwned(u8, alloc, buf, needle, repl);
    alloc.free(buf);
    return out;
}

pub fn main() !void {
    // -------------------------------------------------------------------------
    // Allocator setup with leak check (works in Zig 0.15.x)
    // -------------------------------------------------------------------------
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak)
            std.debug.panic("memory leak detected in rewrite_sections", .{});
    }
    const alloc = gpa.allocator();

    // -------------------------------------------------------------------------
    // Argument parsing
    // -------------------------------------------------------------------------
    var it = try std.process.argsWithAllocator(alloc);
    defer it.deinit();

    _ = it.next(); // exe name
    const path = it.next() orelse {
        std.log.err("usage: rewrite_sections <path-to-static_parameters.zig>", .{});
        return error.MissingPath;
    };

    // -------------------------------------------------------------------------
    // Read file into memory
    // -------------------------------------------------------------------------
    var text = try std.fs.cwd().readFileAlloc(alloc, path, 10 * 1024 * 1024);
    defer alloc.free(text);

    // -------------------------------------------------------------------------
    // Replace Mach-O section names with ELF ones
    // -------------------------------------------------------------------------
    text = try replaceAndFree(alloc, text, "__DATA,__flash_weights", ".flash_weights");
    text = try replaceAndFree(alloc, text, "__TEXT,__const", ".rodata");

    // -------------------------------------------------------------------------
    // Overwrite the original file with the fixed content
    // -------------------------------------------------------------------------
    try std.fs.cwd().writeFile(.{ .sub_path = path, .data = text });

    std.log.info("rewrite_sections: updated {s}", .{path});
}
