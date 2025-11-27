const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
const cg_v1 = @import("../codegen_v1.zig");

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

pub inline fn write_ino(writer: *std.Io.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    _ = linearizedGraph;
    try writer.print("#include <Arduino.h> \n #include <lib_zant.h> \n", .{});
}
