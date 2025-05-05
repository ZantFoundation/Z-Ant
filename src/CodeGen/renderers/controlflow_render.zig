const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    // Discard allocator if not needed
    // _ = allocator;
    if (uop.op != .RANGE and uop.op != .ENDRANGE) {
        return error.InvalidOperation;
    }

    if ((uop.op == .RANGE and uop.src.len != 0) or (uop.op == .ENDRANGE and uop.src.len != 1)) {
        return error.InvalidOperandCount;
    }

    switch (uop.op) {
        .RANGE => {
            const index_name = ptr_map.get(uop.id) orelse return error.VariableNotFound;
            const type_str = DTypeInfo.asString(uop.dtype);
            const range_start = try std.fmt.allocPrint(allocator, "{d}", .{uop.arg.?.loop_bounds.start});
            const range_end = try std.fmt.allocPrint(allocator, "{d}", .{uop.arg.?.loop_bounds.end});
            defer allocator.free(range_start);
            defer allocator.free(range_end);
            try writer.print("var {s}: {s} = {s}; // RANGE (uop {d})\nwhile ({s} < {s}) : ({s} += 1) {{\n", .{ index_name, type_str, range_start, uop.id, index_name, range_end, index_name });
        },
        .ENDRANGE => {
            try writer.print("}}\n", .{});
        },
        else => unreachable,
    }
}
