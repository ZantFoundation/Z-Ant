const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .ADD and uop.op != .SUB and uop.op != .MUL and uop.op != .FDIV and uop.op != .POW) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }

    // Generate the variable names for result and operands
    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);

    const lhs_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    defer allocator.free(lhs_var);

    const rhs_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[1]});
    defer allocator.free(rhs_var);

    const type_str = DTypeInfo.asString(uop.dtype);

    switch (uop.op) {
        .ADD  => try writer.print("const {s} = @as({s}, {s}) + @as({s}, {s});\n", .{ result_var, type_str, lhs_var, type_str, rhs_var }),
        .SUB  => try writer.print("const {s} = @as({s}, {s}) - @as({s}, {s});\n", .{ result_var, type_str, lhs_var, type_str, rhs_var }),
        .MUL  => try writer.print("const {s} = @as({s}, {s}) * @as({s}, {s});\n", .{ result_var, type_str, lhs_var, type_str, rhs_var }),
        .FDIV => try writer.print("const {s} = @as({s}, {s}) / @as({s}, {s});\n", .{ result_var, type_str, lhs_var, type_str, rhs_var }),
        .POW  => try writer.print("const {s} = std.math.pow({s}, {s}, {s});\n",   .{ result_var, type_str, lhs_var, rhs_var }),
        else => unreachable,
    }
}
