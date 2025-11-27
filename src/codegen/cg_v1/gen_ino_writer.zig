const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
const cg_v1 = @import("codegen_v1.zig");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;

// --- utils
pub const utils = IR_zant.utils;
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;

// --- ino writer
const codegenInoFile = @import("inofile/gen_ino.zig");

pub fn write(model_name: []const u8, codegen_parameters: cg_v1.CodegenParameters, linearizedGraph: std.ArrayList(*NodeZant)) !void {

    //The ino.file will be in his model folder
    const ino_file_path = try std.fmt.allocPrint(allocator, "generated/{s}/{s}.ino", .{ model_name, model_name });
    defer allocator.free(ino_file_path);

    var ino_file = try std.fs.cwd().createFile(ino_file_path, .{});

    std.log.info("\n .ino file created, path:{s}", .{ino_file_path});
    defer ino_file.close();

    var ino_file_buffer: [4096]u8 = undefined;
    //create ino.file writer
    var ino_writer = ino_file.writer(&ino_file_buffer);
    const writer = &ino_writer.interface;

    //TODO  far arrivare qui e passare i parametri per il XIP chec
    _ = codegen_parameters;

    // Generate tensor initialization code in the static_parameters.zig file
    try codegenInoFile.write_ino(writer, linearizedGraph); //TODO find ount what parameters it strictly needs

    try writer.flush();
}
