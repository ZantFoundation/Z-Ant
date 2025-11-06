const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;
const pattern_matcher = IR.pattern_matcher;
const pattern_collection = IR.pattern_collection;

// --- Static memory planning
pub const MemoryPlanner = @import("memory_planner/memory_planner.zig");
pub const MemoryPlannerInterface = MemoryPlanner.Interface;
pub const MemoryPlannerTypes = MemoryPlanner.types;
pub const Greedy = MemoryPlanner.Greedy;
pub const IntervalBased = MemoryPlanner.IntervalBased;

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

    var backing_buffers: ?MemoryPlannerTypes.TensorBackingBuffers = null;
    defer {
        if (backing_buffers) |*allocators| {
            allocators.deinit();
        }
    }

    if (!codegen_options.dynamic and codegen_options.static_planning) {
        var interval_based_planner = IntervalBased.init(allocator);
        var memory_planner = MemoryPlannerInterface.init(&interval_based_planner);

        // var greedy_planner = Greedy.init(allocator);
        // var memory_planner = MemoryPlannerInterface.init(&greedy_planner);

        // NOTE: Not a strict requirement for the future, but the first draft
        // will assume that there are no cycles (simplifies the implementation
        // and works for non-recurrent neural networks)
        std.debug.assert(try graphZant.isDag(allocator));
        std.debug.assert(linearizedGraph.items.len > 0);

        backing_buffers = try memory_planner.compute(linearizedGraph.items[0]);

        std.debug.print("\nStatic memory planning", .{});
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        var entry_it = backing_buffers.?.iterator();
        var tensors = try arena_alloc.alloc(struct {
            name: []const u8,
            size: usize,
            backing_buffer: ?MemoryPlannerTypes.BackingBuffer,
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

        const file = try std.fs.cwd().createFile(
            try std.fmt.allocPrint(allocator, "{s}memory_plan.json", .{generated_path}),
            .{},
        );
        defer file.close();
        var file_buffer: [1024]u8 = undefined;
        var file_writer = file.writer(&file_buffer);
        var writer = &file_writer.interface;
        try writer.writeAll(json_str);

        // std.debug.print("\n{s}\n", .{json_str}); // DEBUG
        // std.debug.print("\n", .{});
    }

    try codegnenerateFromLinearizedGraph(
        model_name,
        generated_path,
        linearizedGraph,
        .{ .tensors_backing_buffers = backing_buffers },
    );
}

pub const CodegenParameters = struct {
    tensors_backing_buffers: ?MemoryPlannerTypes.TensorBackingBuffers = null,
};

pub fn codegnenerateFromLinearizedGraph(
    model_name: []const u8,
    generated_path: []const u8,
    linearizedGraph: std.ArrayList(*NodeZant),
    codegen_parameters: CodegenParameters,
) !void {

    //set globals
    tensorZantMap = &IR.tensorZant_lib.tensorMap;

    try ParametersWriter.write(generated_path);

    try PredictWriter.write(generated_path, model_name, linearizedGraph, codegen_parameters);
}
