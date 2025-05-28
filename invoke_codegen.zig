const std = @import("std");
const childProcess = std.process.Child;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const folder_path = "generated/";

    var argv1 = [_][]const u8{ "zig", "build", "codegen" };
    var child1 = childProcess.init(&argv1, allocator);

    var argv2 = [_][]const u8{ "open", folder_path };
    var child2 = childProcess.init(&argv2, allocator);

    try child1.spawn();
    try child2.spawn();
}
