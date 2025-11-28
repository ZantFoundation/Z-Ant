const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
const cg_v1 = @import("codegen_v1.zig");
pub const codegen_options = @import("codegen_options");
const Ino_helper = @import("inofile/ino_helper.zig").Ino_helper;

// --- zant IR
const IR_utils = IR_zant.utils;
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const NodeZant = IR_zant.NodeZant;
const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;

// --- allocator
const allocator = zant.utils.allocator.allocator;

// --- ino writer
const codegenInoFile = @import("inofile/gen_ino.zig");

//
//--- Finds the correct path to generate the .ino file, creates it and writes the content
//
pub fn write(model_name: []const u8) !void {

    // Use the generated path from options, compatible with the generated_path flag
    const ino_file_path = try std.fmt.allocPrint(allocator, "{s}{s}.ino", .{ codegen_options.generated_path, model_name });
    defer allocator.free(ino_file_path);

    var ino_file = try std.fs.cwd().createFile(ino_file_path, .{});

    std.log.info("\n .ino file created, path:{s}", .{ino_file_path});
    defer ino_file.close();

    var ino_file_buffer: [4096]u8 = undefined;

    //create ino.file writer
    var ino_writer = ino_file.writer(&ino_file_buffer);
    const writer = &ino_writer.interface;

    const ino_helper: Ino_helper = try Ino_helper.init();

    try codegenInoFile.write_ino_file(writer, ino_helper);

    try writer.flush();
}
