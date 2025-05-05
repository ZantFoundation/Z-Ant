const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .RANGE and uop.op != .ENDRANGE) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }
    
    const index_name = try std.fmt.allocPrint(allocator, "t{d}", .{ uop.src[0] });
    defer allocator.free(index_name);

    switch(uop.op){
        .RANGE => {
            const type_str = DTypeInfo.asString(uop.dtype);
            const range_start = try std.fmt.allocPrint(allocator, "@as({s},{d})", .{ type_str, uop.arg.?.loop_bounds.start });
            const range_end   = try std.fmt.allocPrint(allocator, "@as({s},{d})", .{ type_str, uop.arg.?.loop_bounds.end });
            defer allocator.free(range_start);
            defer allocator.free(range_end);
            try writer.print("for({s}..{s})|{s}|{{\n", .{range_start, range_end, index_name});
        },
        .ENDRANGE => try writer.print("}} //ending range from id {s}\n", .{ index_name }),
        else => unreachable,
    }

}
