const std = @import("std");
const onnx = @import("IR_zant").onnx;
const IR_zant = @import("IR_zant");
const codegen = @import("codegen");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;

// --- utils
pub const utils = IR_zant.utils;
// --- onnx
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = @import("codegen").pkg_allocator.allocator;

// --- codegen
const codeGenPredict = @import("predict/predict.zig");
const codegen_options = @import("codegen").codegen_options;

pub fn write(
    generated_path: []const u8,
    model_name: []const u8,
    linearizedGraph: std.ArrayList(*NodeZant),
    codegen_parameters: codegen.CodegenParameters,
) !void {
    //initializing writer for lib_operation file
    const lib_file_path = try std.fmt.allocPrint(allocator, "{s}lib_{s}.zig", .{ generated_path, model_name });
    defer allocator.free(lib_file_path);

    var lib_file = try std.fs.cwd().createFile(lib_file_path, .{});

    std.log.info("\n .......... file created, path:{s}", .{lib_file_path});
    defer lib_file.close();

    var lib_file_buffer: [65536]u8 = undefined;
    var lib_file_writer = lib_file.writer(&lib_file_buffer);
    const writer = &lib_file_writer.interface;

    // Write the necessary library imports to the generated Zig file
    try write_libraries(writer);

    // Always write allocation tracking (needed for last_result_size)
    try write_allocationTracking(writer);

    if (codegen_options.log) {
        //log function setting
        try write_logFunction(writer);
    }

    //Fixed Buffer Allocator (only for static allocation)
    if (!codegen_options.dynamic) {
        try write_FBA(writer);
    }

    // _ = linearizedGraph;
    // Generate prediction function code
    try codeGenPredict.writePredict(writer, linearizedGraph, codegen_options.do_export, codegen_parameters);

    try writer.flush();
}

/// Writes the required library imports to the generated Zig file for predict function.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const IR_zant = @import("IR_zant");
        \\ const Tensor = IR_zant.core.tensor.Tensor;
        \\ const tensMath = IR_zant.core.math_standard;
        \\ const pkgAllocator = IR_zant.pkg_allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const utils = @import("codegen").utils;
        \\ const param_lib = @import("static_parameters.zig");
        \\
    , .{});
}

fn write_allocationTracking(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\
        \\// Global allocation tracking for safe deallocation
        \\var last_result_size: usize = 0;
        \\
        \\// Deallocator function for external C usage
        \\pub {s} fn zant_free_result(ptr: ?[*]T_out) callconv(.c) void {{
        \\    if (ptr) |valid_ptr| {{
        \\        if (last_result_size > 0) {{
        \\            const slice = valid_ptr[0..last_result_size];
        \\            allocator.free(slice);
        \\            last_result_size = 0;
        \\        }}
        \\    }}
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}

fn write_logFunction(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]u8) callconv(.c) void = null;
        \\
        \\pub {s} fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.c) void) void {{
        \\    log_function = func;
        \\    // Forward to core so ops (e.g., qlinearconv) can log via same callback
        \\    zant.core.tensor.setLogFunction(func);
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}

fn write_FBA(writer: *std.Io.Writer) !void {
    const buffer_size_kb = if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZANT_FBA_SIZE_KB")) |env_size| blk: {
        defer std.heap.page_allocator.free(env_size);
        break :blk std.fmt.parseInt(u32, env_size, 10) catch 512;
    } else |_| 1024; // Default 1MB to handle peak allocation

    var link_section: ?[]u8 = null;
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZANT_FBA_SECTION")) |section_name| {
        link_section = section_name;
    } else |_| {}

    // Use tensor_pool if the option is enabled or if custom section is specified
    const should_use_tensor_pool = codegen_options.use_tensor_pool or link_section != null;

    const old_static_format =
        \\
        \\
        \\ // Static allocation: two FixedBufferAllocator pools (ping-pong)
        \\ // Buffer size: {[buffer_size_kb]d}KB each (configurable via ZANT_FBA_SIZE_KB env var)
        \\ var buf_a: [{[buffer_size_bytes]d}]u8 {[link_section]s} = undefined;
        \\ var fba_state_a = std.heap.FixedBufferAllocator.init(&buf_a);
        \\ const fba_a = fba_state_a.allocator();
        \\ var fba_live_a: usize = 0; // live LINK tensors in pool A
        \\
        \\ var buf_b: [{[buffer_size_bytes]d}]u8 {[link_section]s} = undefined;
        \\ var fba_state_b = std.heap.FixedBufferAllocator.init(&buf_b);
        \\ const fba_b = fba_state_b.allocator();
        \\ var fba_live_b: usize = 0; // live LINK tensors in pool B
        \\ const fba = fba_a; // Backward compatibility path
        \\
        \\
    ;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const arena_alloc = arena.allocator();

    if (!codegen_options.static_planning) {
        const section = link_section orelse ".tensor_pool";
        try writer.print(old_static_format, .{
            .link_section = if (should_use_tensor_pool) blk: {
                break :blk try std.fmt.allocPrint(arena_alloc, "linksection(\"{s}\")", .{section});
            } else blk: {
                break :blk "";
            },
            .buffer_size_kb = buffer_size_kb,
            .buffer_size_bytes = buffer_size_kb * 1024,
        });
        if (link_section) |section_to_free| {
            std.heap.page_allocator.free(section_to_free);
        }
    } else {
        // We still emit a FBA with a zero-sized backing buffer
        // Why? Tensors still need an allocator argument, but they don't use it
        // when initialized with fromConstBuffer
        // TODO: consider making the tensor allocator optional
        try writer.print(
            \\
            \\ // Static allocation: placeholder FixedBufferAllocator
            \\ // Tensors still need an allocator argument, but they don't use it
            \\ // when initialized with fromConstBuffer
            \\
            \\ var buf_a: [0]u8 = undefined;
            \\ var fba_state_a = std.heap.FixedBufferAllocator.init(&buf_a);
            \\ const fba = fba_state_a.allocator();
            \\
        , .{});
    }
}
