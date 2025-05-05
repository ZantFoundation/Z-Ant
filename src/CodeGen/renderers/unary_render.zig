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
    _ = allocator;
    if (uop.op != .EXP2 and uop.op != .NEG and uop.op != .CAST) {
        return error.InvalidOperation;
    }

    //minimal Max and min are only attainable from two variables, not more or less.
    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }

    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const src_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;

    // const result_type_str = DTypeInfo.asString(uop.dtype);

    // const first_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    // defer allocator.free(first_var);

    // Need source type for CAST
    // const src_type_str = DTypeInfo.asString(???); // How to get src uop's type?

    // Updated print statements to assign to existing buffer elements (assuming scalar ops for now)
    switch (uop.op) {
        .EXP2 => try writer.print("{s}[0] = std.math.exp2({s}[0]); // EXP2 (uop {d})\n", .{ result_var, src_var, uop.id }),
        .NEG => try writer.print("{s}[0] = -{s}[0]; // NEG (uop {d})\n", .{ result_var, src_var, uop.id }),
        .CAST => {
            @panic("CAST rendering needs source type information");
        },
        else => unreachable,
    }
}
