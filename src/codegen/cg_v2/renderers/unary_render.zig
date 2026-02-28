const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    const supported_ops = [_]UOpType{ .EXP2, .NEG, .CAST, .CLIP };

    // Validate operation type
    if (!std.mem.containsAtLeast(UOpType, &supported_ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }

    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const src_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;

    switch (uop.op) {
        .EXP2 => {
            const type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("    const {s}: {s} = std.math.exp2({s}); // EXP2 (uop {d})\n", .{ result_var, type_str, src_var, uop.id });
            try writer.print("    _ = &{s};\n", .{result_var});
        },
        .NEG => {
            const type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("    const {s}: {s} = -{s}; // NEG (uop {d})\n", .{ result_var, type_str, src_var, uop.id });
            try writer.print("    _ = &{s};\n", .{result_var});
        },
        .CAST => {
            //CAST RENDERING NEEDS SOURCE TYPE INFORMATION
            return error.NotImplemented;
        },
        .CLIP => {
            const t: usize = 12123;
            const t8: u8 = t & 0xFF;
            _ = t8;
            if (uop.dtype != uop.arg.?.clip_bounds.type) {
                return error.TypeNotMatching;
            }

            if (uop.arg.?.clip_bounds.max.getDType() != uop.arg.?.clip_bounds.type or
                uop.arg.?.clip_bounds.min.getDType() != uop.arg.?.clip_bounds.type)
            {
                return error.TypeNotMatching;
            }

            const is_accumulator_update = std.mem.startsWith(u8, src_var, "acc_");

            const maxstr = switch (uop.arg.?.clip_bounds.type) {
                .f32 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.max.f32}),
                .i32 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.max.i32}),
                .i8 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.max.i8}),
                .u16 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.max.u16}),
                else => return error.ClipWithBoolean,
            };
            defer allocator.free(maxstr);

            const minstr = switch (uop.arg.?.clip_bounds.type) {
                .f32 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.min.f32}),
                .i32 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.min.i32}),
                .i8 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.min.i8}),
                .u16 => try std.fmt.allocPrint(allocator, "{}", .{uop.arg.?.clip_bounds.min.u16}),
                else => return error.ClipWithBoolean,
            };
            defer allocator.free(minstr);

            if (is_accumulator_update) {
                try writer.print("  {s} = if ( {s} < {s} ) if ( {s} > {s} ) {s} else {s} else {s};\n", .{ src_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr });
            } else {
                try writer.print("  const {s} = if ( {s} < {s} ) if ( {s} > {s} ) {s} else {s} else {s};\n", .{ result_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr });
            }
        },
        else => unreachable,
    }
}

// if(is_accumulator_update){
//     try writer.print(" {s} = if({s} < {s}) { if ({s} > {s}) {s} else {s};} else {s}; // CLIP into Accumulator (uop {d})\n",
//         .{result_var, src_var, min_str, src_var, max_str, max_str, src_var, min_str, uop.id});
// } else {
//     try writer.print(" const {s} = if({s} < {s}) { if ({s} > {s}) {s} else {s};} else {s};\n",
//         .{result_var, src_var, min_str, src_var, max_str, max_str, src_var, min_str});
// }
