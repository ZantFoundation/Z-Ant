const std = @import("std");

// Return codes standardized - matching existing behavior
pub const RC = struct {
    pub const OK: i32 = 0;
    pub const MATH_ERROR: i32 = -1;
    pub const INIT_ERROR: i32 = -2;
    pub const RETURN_ERROR: i32 = -3;
};

// Centralized logging helper
pub fn emitLogHelper(writer: std.fs.File.Writer) !void {
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
pub fn emitFunctionSignature(writer: std.fs.File.Writer, do_export: bool) !void {
    try writer.print(
        \\
        \\ // return codes:
        \\ //  0 : everything good
        \\ // -1 : something went wrong in the mathematical operations
        \\ // -2 : something went wrong in the initialization phase
        \\ // -3 : something went wrong in the output/return phase
        \\pub {s} fn predict (
        \\    input: [*]T_in,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T_out,
        \\) {s} i32 {{
    , .{
        if (do_export) "export" else "",
        if (do_export) "callconv(.C)" else "",
    });
}

// Helper for input size calculation - extracted from common pattern
pub fn emitInputSizeCalculation(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\  
        \\    //computing the size of the input tensor
        \\    var size: u32 = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        size *= input_shape[dim_i];
        \\    }}
        \\     
        \\    //allocating space in memory for the data
        \\    const data = allocator.alloc(T_in, size) catch return {d};
        \\    defer allocator.free(data);
        \\    for (0..size) |i| {{
        \\        data[i] = input[i]; // Copying input elements 
        \\    }}
        \\    
        \\    //converting the shape from [*]u32 to []usize
        \\    const usized_shape: []usize = utils.u32ToUsize(allocator, input_shape, shape_len) catch return {d};
    , .{ RC.INIT_ERROR, RC.INIT_ERROR });
}
