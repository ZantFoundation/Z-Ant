const std = @import("std");

// Return codes standardized - matching existing behavior
pub const RC = struct {
    pub const OK: i32 = 0;
    pub const MATH_ERROR: i32 = -1;
    pub const INIT_ERROR: i32 = -2;
    pub const RETURN_ERROR: i32 = -3;
};

// Centralized logging helper
pub fn emitLogHelper(writer: *std.Io.Writer) !void {
    try writer.print(
        \\
        \\inline fn logMsg(comptime msg: []const u8) void {{
        \\    if (log_function) |log| {{
        \\        log(@constCast(@ptrCast(msg)));
        \\    }}
        \\}}
        \\
    , .{});
}

// Standard function signature template
pub fn emitFunctionSignature(writer: *std.Io.Writer, do_export: bool) !void {
    try writer.print(
        \\
        \\ // return codes:
        \\ //  0 : everything good
        \\ // -2 : something went wrong in the initialization phase
        \\ // -3 : something went wrong in the output/return phase
        \\ // -100 onwards : something went wrong in the mathematical operations, the specific error 
        \\ //              code indicates which operation failed (e.g., -101 means first operation failed, -102 second, etc.)
        \\pub {s} fn predict (
        \\    input: [*]T_in,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T_out,
        \\) {s} i32 {{
    , .{
        if (do_export) "export" else "",
        if (do_export) "callconv(.c)" else "",
    });
}

// Helper for input size calculation - extracted from common pattern
pub fn emitInputSizeCalculation(writer: *std.Io.Writer) !void {
    try writer.print(
        \\  
        \\    //computing the size of the input tensor (runtime)
        \\    var input_size: usize = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        input_size *= @as(usize, input_shape[dim_i]);
        \\    }}
    , .{});
}
