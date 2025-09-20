const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;
const pattern_matcher = IR.pattern_matcher;
const pattern_collection = IR.pattern_collection;

// --- utils
pub const utils = @import("utils.zig");
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;
// -- writers
const ParametersWriter = @import("parameter_writer.zig");
const PredictWriter = @import("predict_writer.zig");

pub const codegen_options = @import("codegen_options");

// -- testing
pub const testWriter = @import("tests_writer.zig");

// -- GLOBAL VARIABLES
pub var tensorZantMap: *std.StringHashMap(TensorZant) = undefined;

pub fn codegnenerateFromOnnx(model_name: []const u8, generated_path: []const u8, model: ModelOnnx) !void {

    // Create the generated model directory if not present
    try std.fs.cwd().makePath(generated_path);

    //create the Zant Intermediate Representation
    var graphZant: GraphZant = try IR.init(@constCast(&model));
    defer graphZant.deinit();

    try codegnenerateFromGraphZant(model_name, generated_path, &graphZant);
}

pub fn codegnenerateFromGraphZant(model_name: []const u8, generated_path: []const u8, graphZant: *GraphZant) !void {
    var linearizedGraph_prefusion: std.ArrayList(*NodeZant) = try graphZant.linearize(allocator);
    defer linearizedGraph_prefusion.deinit();

    // --- fusion step ---
    if (codegen_options.fuse) try graphZant.fuse(&pattern_collection.patterns);

    // graphZant.print_before_linearizzation(); // DEBUG

    // linearizing the fused graph
    var linearizedGraph: std.ArrayList(*NodeZant) = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit();

    std.debug.print("\n\n--- Linearized Graph pre fusion: ", .{});
    for (linearizedGraph_prefusion.items) |node| {
        std.debug.print("\n  {s} ", .{node.name.?});
    }
    std.debug.print("\n\n --- Linearized Graph post fusion : ", .{});
    for (linearizedGraph.items) |node| {
        std.debug.print("\n  {s} ", .{node.name.?});
    }
    std.debug.print("\n", .{});

    std.debug.print("\n Pre-Fusion nodes: {} \n Post-Fusion nodes: {}", .{ linearizedGraph_prefusion.items.len, linearizedGraph.items.len });

    try codegnenerateFromLinearizedGraph(model_name, generated_path, linearizedGraph);
}

pub fn codegnenerateFromLinearizedGraph(model_name: []const u8, generated_path: []const u8, linearizedGraph: std.ArrayList(*NodeZant)) !void {

    //set globals
    tensorZantMap = &IR.tensorZant_lib.tensorMap;

    try ParametersWriter.write(generated_path);

    try PredictWriter.write(generated_path, model_name, linearizedGraph);
}
