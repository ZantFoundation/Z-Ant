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
    if (uop.op != .ADD and uop.op != .SUB and uop.op != .MUL and uop.op != .FDIV and uop.op != .POW and uop.op != .MAX and uop.op != .MIN and uop.op != .CMPLT) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }

    // Get variable names from ptr_map
    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const lhs_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;
    const rhs_var = ptr_map.get(uop.src[1]) orelse return error.VariableNotFound;

    const type_str = DTypeInfo.asString(uop.dtype);

    switch (uop.op) {
        .ADD => try writer.print("{s}[0] = {s}[0] + {s}[0]; // ADD (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .SUB => try writer.print("{s}[0] = {s}[0] - {s}[0]; // SUB (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .MUL => try writer.print("{s}[0] = {s}[0] * {s}[0]; // MUL (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .FDIV => try writer.print("{s}[0] = {s}[0] / {s}[0]; // FDIV (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .POW => try writer.print("{s}[0] = std.math.pow({s}, {s}[0], {s}[0]); // POW (uop {d})\n", .{ result_var, type_str, lhs_var, rhs_var, uop.id }),
        .MAX => try writer.print("{s}[0] = @max({s}[0], {s}[0]); // MAX (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .MIN => try writer.print("{s}[0] = @min({s}[0], {s}[0]); // MIN (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        .CMPLT => try writer.print("{s}[0] = ({s}[0] < {s}[0]); // CMPLT (uop {d})\n", .{ result_var, lhs_var, rhs_var, uop.id }),
        else => unreachable,
    }
}
