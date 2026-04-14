const std = @import("std");
const onnx = @import("IR_zant").onnx;
const IR_zant = @import("IR_zant");
pub const codegen_options = @import("codegen_options");
const H_helper = @import("hfile/h_helper.zig").H_helper;

// --- zant IR
const IR_utils = IR_zant.utils;
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const NodeZant = IR_zant.NodeZant;
const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

// --- onnx
const ModelOnnx = onnx.ModelProto;

// --- allocator
const allocator = @import("zant_utils").allocator.allocator;

// --- h writer
const codegenHFile = @import("hfile/gen_h.zig");

//
//--- Finds the correct path to generate the .H file, creates it and writes the content
//
pub fn write() !void {

    // Use the generated path from options, compatible with the generated_path flag
    const h_file_path = try std.fmt.allocPrint(allocator, "{s}lib_zant.h", .{codegen_options.generated_path});
    defer allocator.free(h_file_path);

    var h_file = try std.fs.cwd().createFile(h_file_path, .{});

    std.log.info("\n .h file created, path:{s}", .{h_file_path});
    defer h_file.close();

    var h_file_buffer: [65536]u8 = undefined;

    //create .h file writer
    var h_writer = h_file.writer(&h_file_buffer);
    const writer = &h_writer.interface;

    const h_helper: H_helper = try H_helper.init();

    try codegenHFile.write_h_file(writer, h_helper);

    try writer.flush();
}
