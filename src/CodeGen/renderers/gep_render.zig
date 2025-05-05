const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;
const ViewInfo = @import("view_manager.zig").ViewInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: std.AutoHashMap(usize, ViewInfo),
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    if (uop.op != .GEP) {
        return error.InvalidOperation;
    }
    if (uop.arg == null) {
        return error.NoAnyProvided;
    }

    const mem_info = uop.arg.?.mem_info;
    const offset = mem_info.offset;
    const stride = mem_info.stride;
    const type_str = DTypeInfo.asString(uop.dtype);
    const result_addr_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;

    std.debug.print("DEBUG: GepRender looking up base_uop_id: {d} in ptr_map\n", .{uop.src[0]});
    const base_uop_id = uop.src[0];
    const base_ptr_expr = ptr_map.get(base_uop_id) orelse {
        std.debug.print("GEP Error: Base ID {d} not found in ptr_map\n", .{base_uop_id});
        return error.InvalidOperation;
    };

    var index_expr_str: []const u8 = undefined;
    var temp_alloc = std.ArrayList(u8).init(allocator);
    defer temp_alloc.deinit();
    var needs_free_index_expr = false;

    const view_id = uop.src[0];
    if (view_map.get(view_id)) |view_info| {
        if (uop.src.len == 2) {
            const index_var_id = uop.src[1];
            index_expr_str = ptr_map.get(index_var_id) orelse return error.VariableNotFound;
            needs_free_index_expr = false;
        } else if (uop.src.len > 2 and uop.src.len - 1 == view_info.arg.view_meta.strides.len) {
            const strides = view_info.arg.view_meta.strides;
            const indexes = uop.src;
            var first_term = true;

            var i: usize = 1;
            while (i < indexes.len) : (i += 1) {
                const index_var_id = indexes[i];
                const view_stride_val = strides[i - 1];
                const index_var_name = ptr_map.get(index_var_id) orelse return error.VariableNotFound;

                if (view_stride_val != 0) {
                    if (!first_term) try temp_alloc.writer().print(" + ", .{});
                    try temp_alloc.writer().print("({s} * {d})", .{ index_var_name, view_stride_val });
                    first_term = false;
                }
            }
            if (first_term) try temp_alloc.writer().print("0", .{});
            index_expr_str = temp_alloc.items;
            needs_free_index_expr = false;
        } else {
            std.debug.print("GEP Error: src.len ({d}) mismatch with view strides.len ({d}) for view_id {d}\n", .{ uop.src.len, view_info.arg.view_meta.strides.len, view_id });
            return error.InvalidOperation;
        }
    } else {
        if (uop.src.len != 2) {
            std.debug.print("GEP Error: Non-view GEP expects src.len == 2, got {d}\n", .{uop.src.len});
            return error.InvalidOperation;
        }
        const index_var_id = uop.src[1];
        index_expr_str = ptr_map.get(index_var_id) orelse return error.VariableNotFound;
        needs_free_index_expr = false;
    }

    // Determine the base pointer expression (add .ptr for slices)
    var base_ptr_final_expr_buf: [128]u8 = undefined;
    var base_ptr_final_expr: []const u8 = base_ptr_expr;
    if (std.mem.startsWith(u8, base_ptr_expr, "input_") or
        std.mem.startsWith(u8, base_ptr_expr, "output_"))
    {
        base_ptr_final_expr = try std.fmt.bufPrint(&base_ptr_final_expr_buf, "{s}.ptr", .{base_ptr_expr});
    }

    // Declare the result address variable using the name from ptr_map
    // Cast the index expression ({s}) to usize before multiplying by stride ({d})
    try writer.print("const {s} = @intFromPtr({s}) + (({d}) + (@as(usize, @intCast({s}))) * {d}) * @sizeOf({s}); // GEP (uop {d})\n", .{
        result_addr_var, // Use name from ptr_map
        base_ptr_final_expr, // Use potentially modified expression (.ptr added)
        offset,
        index_expr_str, // This is the index expression (e.g., "idx_3")
        stride,
        type_str,
        uop.id,
    });

    if (needs_free_index_expr) {
        allocator.free(index_expr_str);
    }
}
