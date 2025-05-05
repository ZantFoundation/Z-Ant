const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DType = zant.uops.DType;
const Any = zant.uops.Any;

pub const ViewInfo = struct {
    dtype: DType,
    src: []const usize,
    arg: Any,
};

pub fn manage(uop: UOp, view_map: *std.AutoHashMap(usize, ViewInfo)) !void {
    if (uop.arg == null) {
        return error.NoAnyProvided;
    }

    const view_info: ViewInfo = .{ .dtype = uop.dtype, .src = uop.src, .arg = uop.arg.? };
    const view_id = uop.id;
    try view_map.put(view_id, view_info);

    std.debug.print("Value: {d} managed with view ID {d}\n", .{ view_id, view_info.src[0] });
}
