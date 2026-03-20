const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen_v2.zig");
const IR_zant = @import("IR_zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const TensorProto = onnx.TensorProto;
const allocator = zant.utils.allocator.allocator;

// Access global codegen state and utilities
const globals = codegen.globals;
const utils = codegen.utils;
const codeGenInitializers = codegen.parameters;
const coddeGenPredict = codegen.predict;
const codegen_options = @import("codegen_options");
const NodeZant = IR_zant.NodeZant;

/// Writes a Zig source file containing the generated code for an ONNX model.
///
/// This function generates the necessary Zig code to initialize tensors and
/// define the prediction logic based on the given ONNX model.
///
/// # Parameters
/// - `file`: The file where the generated Zig code will be written.
/// - `model`: The ONNX model from which to generate the Zig code.
///
/// # Errors
/// This function may return an error if writing to the file fails.
pub fn writeZigFile(model_name: []const u8, model_path: []const u8, nodes: std.ArrayList(*NodeZant), do_export: bool) !void {

    //initializing writer for lib_operation file
    const lib_file_path = try std.fmt.allocPrint(allocator, "{s}lib_{s}.zig", .{ model_path, model_name });
    defer allocator.free(lib_file_path);
    var lib_file = try std.fs.cwd().createFile(lib_file_path, .{
        .read = false,
        .truncate = true,
    });
    std.log.info("\n .......... file created, path:{s}", .{lib_file_path});
    defer lib_file.close();

    var buffer: [65536]u8 = undefined;
    const lib_writer = lib_file.writer(&buffer).interface;

    //initializing writer for static_parameters file
    const params_file_path = try std.fmt.allocPrint(allocator, "{s}static_parameters.zig", .{model_path});
    defer allocator.free(params_file_path);
    var param_file = try std.fs.cwd().createFile(params_file_path, .{});
    std.log.info("\n .......... file created, path:{s}", .{params_file_path});
    defer param_file.close();

    const param_writer = param_file.writer();

    // Write the necessary library imports to the generated Zig file
    try write_libraries(lib_writer);

    if (codegen_options.log) {
        //log function setting
        try write_logFunction(lib_writer);
    }

    //Fixed Buffer Allocator
    try write_FBA(lib_writer);

    try write_type_T(lib_writer);

    // Generate tensor initialization code in the static_parameters.zig file
    try codeGenInitializers.write_parameters(param_writer);

    // Generate prediction function code
    try coddeGenPredict.writePredict(lib_writer, nodes, do_export);
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
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const tensMath = zant.core.tensor.math_standard;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const codegen = @import("codegen");
        \\ const utils = codegen.utils;
        \\ const param_lib = @import("static_parameters.zig");
        \\
    , .{});
}

fn write_logFunction(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]u8) callconv(.c) void = null;
        \\
        \\pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.c) void) void {{
        \\    log_function = func;
        \\    // Forward to core so ops (e.g., qlinearconv) can log via same callback
        \\    zant.core.tensor.setLogFunction(func);
        \\}}
        \\
    , .{});
}

fn write_FBA(writer: *std.Io.Writer) !void {
    // Select allocator strategy based on flag
    if (codegen_options.dynamic) {
        // Use heap-based dynamic allocation
        try writer.writeAll(
            \\
            \\
            \\ // Dynamic allocation: RawCAllocator
        );
    } else {
        // Use fixed buffer allocator for static allocations
        try writer.writeAll(
            \\
            \\
            \\ // Static allocation: FixedBufferAllocator
            \\ var buf: [4096 * 10]u8 = undefined;
            \\ var fba_state = std.heap.FixedBufferAllocator.init(&buf);
            \\ const fba = fba_state.allocator();
            \\
        );
    }
}

fn write_type_T(writer: *std.Io.Writer) !void {
    // Emit the tensor element type derived from the ONNX model input
    const type_str = try utils.getTypeString(globals.networkInputDataType);
    _ = try writer.print(
        \\
        \\ const T = {s};
    , .{type_str});
}
