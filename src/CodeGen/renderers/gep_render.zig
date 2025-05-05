const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .GEP) {
        return error.InvalidOperation;
    }

    if (uop.arg == null) {
        return error.NoAnyProvided;
    }

    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }

    const mem_info = uop.arg.?.mem_info;

    const base_ptr = mem_info.base;
    const offset = mem_info.offset;
    const stride = mem_info.stride;

    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);

    const first_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    defer allocator.free(first_var);

    const second_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[1]});
    defer allocator.free(second_var);

    const type_str = DTypeInfo.asString(uop.dtype);

    try writer.print("const {s} = {d} + ({d} + ({s}[{s}] * {d})) * @as(usize, @sizeOf({s}));\n", .{
        result_var,
        base_ptr,
        offset,
        first_var,
        second_var,
        stride,
        type_str,
    });
}
