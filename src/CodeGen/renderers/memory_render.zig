const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(
    allocator: std.mem.Allocator, // Needed for allocPrint
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8), // Add ptr_map argument
) !void {
    _ = allocator; // Discard unused allocator
    if (uop.op != .DEFINE_GLOBAL and uop.op != .LOAD and uop.op != .STORE and uop.op != .CONST) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 0 and uop.op == .DEFINE_GLOBAL) {
        return error.InvalidOperandCount;
    }

    if (uop.src.len != 2 and uop.op == .STORE) {
        return error.InvalidOperandCount;
    }

    if (uop.src.len != 1 and uop.op == .LOAD) {
        return error.InvalidOperandCount;
    }

    switch (uop.op) {
        .DEFINE_GLOBAL => {
            return error.InvalidOperation;
        },
        .STORE => {
            if (uop.src.len != 2) return error.InvalidOperandCount;
            const gep_uop_id = uop.src[0];
            const value_uop_id = uop.src[1];
            // Get variable names from ptr_map
            const gep_addr_var = ptr_map.get(gep_uop_id) orelse return error.VariableNotFound; // Or specific error
            const value_var = ptr_map.get(value_uop_id) orelse return error.VariableNotFound; // Or specific error
            // const gep_addr_var = try std.fmt.allocPrint(allocator, "t{d}_addr", .{gep_uop_id});
            // defer allocator.free(gep_addr_var);
            // const value_var = try std.fmt.allocPrint(allocator, "t{d}", .{value_uop_id});
            // defer allocator.free(value_var);
            // Use variable names directly
            // Get the type string for the pointer cast
            const ptr_type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("@as(*{s}, @ptrFromInt({s})).* = {s}[0]; // STORE (uop {d})\n", .{ ptr_type_str, gep_addr_var, value_var, uop.id });
            // Old scalar assumption:
            // try writer.print("{s}[0] = {s}; // STORE (uop {d})\n", .{ gep_addr_var, value_var, uop.id });
            // Original line:
            // try writer.print("@as(*{s}, @ptrFromInt({s})).* = ({s});\n", .{ type_str, gep_addr_var, value_var });
        },
        .LOAD => {
            if (uop.src.len != 1) return error.InvalidOperandCount;
            const gep_uop_id = uop.src[0];
            // Get variable name from ptr_map
            const gep_addr_var = ptr_map.get(gep_uop_id) orelse return error.VariableNotFound; // Or specific error
            // const gep_addr_var = try std.fmt.allocPrint(allocator, "t{d}_addr", .{gep_uop_id});
            // defer allocator.free(gep_addr_var);
            const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound; // Get name for the result
            // const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
            // defer allocator.free(result_var);
            // Use variable names directly
            // Get the type string for the pointer cast
            const ptr_type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("{s}[0] = @as(*const {s}, @ptrFromInt({s})).*; // LOAD (uop {d})\n", .{ result_var, ptr_type_str, gep_addr_var, uop.id });
            // Old scalar assumption:
            // try writer.print("{s}[0] = {s}[0]; // LOAD (uop {d})\n", .{ result_var, gep_addr_var, uop.id });
            // Original line:
            // try writer.print("const {s} = @as(*const {s}, @ptrFromInt({s})).*;\n", .{ result_var, type_str, gep_addr_var });
        },
        // Added CONST handling
        .CONST => {
            const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
            const type_str = DTypeInfo.asString(uop.dtype);

            // Ensure arg exists and get value based on dtype
            const arg = uop.arg orelse return error.NoAnyProvided;
            var val_str_buf: [128]u8 = undefined; // Temporary buffer for formatting
            var val_str: []const u8 = undefined;

            // Access fields on the unwrapped `arg`
            switch (uop.dtype) {
                .f32 => {
                    const value = arg.float; // Access .float on unwrapped arg
                    val_str = try std.fmt.bufPrint(&val_str_buf, "{e}", .{value}); // Use {e} for float
                },
                .i32 => {
                    const value = arg.int; // Access .int on unwrapped arg (assuming i32 maps to Any.int)
                    val_str = try std.fmt.bufPrint(&val_str_buf, "{d}", .{value}); // Use {d} for int
                },
                // Add other types as needed
                else => return error.UnsupportedDType,
            }

            // This allocPrint/defer logic is incorrect if using bufPrint
            // const val_str = switch (uop.dtype) {
            //     .f32 => |v| std.fmt.allocPrint(allocator, "{d:.?}", .{v}), // Format float
            //     .i32 => |v| std.fmt.allocPrint(allocator, "{d}", .{v}), // Format int
            //     // Add other types as needed
            //     else => return error.UnsupportedDType,
            // };
            // defer allocator.free(val_str);

            // Assign the formatted value from bufPrint
            try writer.print("{s}[0] = ({s})({s}); // CONST (uop {d})\n", .{ result_var, type_str, val_str, uop.id });
        },
        else => unreachable,
    }
}
