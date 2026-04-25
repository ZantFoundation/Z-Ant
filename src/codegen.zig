//! Codegen v1 — main code-generation pipeline (currently the only active backend).
//!
//! Converts a Zant IR graph into a self-contained C/Arduino inference library.
//! Pipeline: ONNX model → `GraphZant` → optional kernel fusion → linearised
//! node list → static memory planning → file writers (parameters, predict, .ino, .h).
//!
//! Public entry points (in dependency order):
//! - `codeGenerateFromOnnx`            – full pipeline starting from a raw `ModelProto`.
//! - `codeGenerateFromGraphZant`        – pipeline starting from an already-built IR graph.
//! - `codeGenerateFromLinearizedGraph`  – pipeline starting from a linearised node list.
const std = @import("std");
const IR = @import("IR_zant");
const onnx = IR.onnx;

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;
const pattern_matcher = IR.pattern_matcher;
const pattern_collection = IR.pattern_collection;

// --- Static memory planning
pub const static_memory_planning = @import("codegen/static_memory_planning.zig");

// --- utils
pub const utils = @import("codegen/utils.zig");
// --- onnx
const ModelOnnx = onnx.ModelProto;
// --- allocator (re-exported from IR_zant so codegen-internal files can reach it via @import("codegen"))
pub const pkg_allocator = IR.pkg_allocator;
const allocator = pkg_allocator.allocator;
// -- writers
const ParametersWriter = @import("codegen/parameter_writer.zig");
const PredictWriter = @import("codegen/predict_writer.zig");
const InoWriter = @import("codegen/gen_ino_writer.zig");
const HWriter = @import("codegen/gen_h_writer.zig");

pub const codegen_options = @import("codegen_options");

// -- testing
pub const testWriter = @import("codegen/tests_writer.zig");

pub fn codeGenerateFromOnnx(model_name: []const u8, generated_path: []const u8, model: ModelOnnx) !void {

    // Create the generated model directory if not present
    try std.fs.cwd().makePath(generated_path);

    //create the Zant Intermediate Representation
    var graphZant: GraphZant = try IR.init(@constCast(&model));
    defer graphZant.deinit();

    try codeGenerateFromGraphZant(model_name, generated_path, &graphZant);
}

pub fn codeGenerateFromGraphZant(model_name: []const u8, generated_path: []const u8, graphZant: *GraphZant) !void {
    const PreFusionNodes = graphZant.nodes.items.len;
    const PreFusion_linkers = (try IR.utils.getLinkers(&IR.tensorZant_lib.tensorMap)).len;

    // --- fusion step ---
    if (codegen_options.fuse) try graphZant.fuse(&pattern_collection.patterns);

    // graphZant.print_before_linearizzation(); // DEBUG

    // Note: Pre-fusion graph printing disabled to avoid accessing freed nodes

    // try graphZant.print_linearized(); // DEBUG

    std.debug.print("\n Pre-Fusion nodes: {} \n Post-Fusion nodes: {}", .{ PreFusionNodes, graphZant.nodes.items.len });

    std.debug.print("\n-----\n Pre-Fusion LINK TENSORS: {} \n Post-Fusion LINK TENSORS: {}\n Post-Fusion FUSED_LINK TENSORS: {}", .{
        PreFusion_linkers,
        (try IR.utils.getLinkers(&IR.tensorZant_lib.tensorMap)).len,
        (try IR.utils.getFusedLinkers(&IR.tensorZant_lib.tensorMap)).len,
    });

    var linearizedGraph: std.ArrayList(*NodeZant) = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit(allocator);

    var backing_buffers: ?static_memory_planning.TensorsBackingBuffers = null;
    defer {
        if (backing_buffers) |*allocators| {
            allocators.deinit();
        }
    }

    if (!codegen_options.dynamic and codegen_options.static_planning) {
        // NOTE: Not a strict requirement for the future, but the first draft
        // will assume that there are no cycles (simplifies the implementation
        // and works for non-recurrent neural networks)
        std.debug.assert(try graphZant.isDag(allocator));
        std.debug.assert(linearizedGraph.items.len > 0);

        backing_buffers = try static_memory_planning.computeBackingBuffers_v0(linearizedGraph.items[0], allocator);

        std.debug.print("\nStatic memory planning", .{});
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        var entry_it = backing_buffers.?.iterator();
        var tensors = try arena_alloc.alloc(struct {
            name: []const u8,
            size: usize,
            backing_buffer: ?static_memory_planning.BackingBuffer,
        }, backing_buffers.?.count());
        var i: usize = 0;
        while (entry_it.next()) |entry| : (i += 1) {
            const tensor = IR.tensorZant_lib.tensorMap.get(entry.key_ptr.*).?;
            tensors[i] = .{
                .name = tensor.name,
                .size = tensor.getSize(),
                .backing_buffer = entry.value_ptr.*,
            };
        }

        var json_writer: std.Io.Writer.Allocating = .init(allocator);
        defer json_writer.deinit();
        const tensors_json = std.json.fmt(tensors, .{});
        try tensors_json.format(&json_writer.writer);
        const json_str = try json_writer.toOwnedSlice();
        defer allocator.free(json_str);

        const plan_file_path = try std.fmt.allocPrint(allocator, "{s}memory_allocation_{s}.json", .{ generated_path, model_name });
        defer allocator.free(plan_file_path);

        var plan_file = try std.fs.cwd().createFile(plan_file_path, .{ .truncate = true });
        defer plan_file.close();
        try plan_file.writeAll(json_str);
    }

    try codeGenerateFromLinearizedGraph(
        model_name,
        generated_path,
        linearizedGraph,
        .{ .tensors_backing_buffers = backing_buffers },
    );
}

pub const CodegenParameters = struct {
    tensors_backing_buffers: ?static_memory_planning.TensorsBackingBuffers = null,
};

pub fn codeGenerateFromLinearizedGraph(
    model_name: []const u8,
    generated_path: []const u8,
    linearizedGraph: std.ArrayList(*NodeZant),
    codegen_parameters: CodegenParameters,
) !void {
    try ParametersWriter.write(generated_path);

    try PredictWriter.write(generated_path, model_name, linearizedGraph, codegen_parameters);

    if (codegen_options.gen_ino) try InoWriter.write(model_name);
    if (codegen_options.gen_h) try HWriter.write();
}
