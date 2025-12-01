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
const Planners = .{ MemoryPlanner.BestFitDefrag, MemoryPlanner.Greedy, MemoryPlanner.IntervalBased };

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
    try std.fs.cwd().makePath(generated_path);
    var graphZant: GraphZant = try IR.init(@constCast(&model));
    defer graphZant.deinit();
    try codegnenerateFromGraphZant(model_name, generated_path, &graphZant);
}

const PlannerResult = struct {
    name: []const u8,
    buffers: ?MemoryPlannerTypes.TensorBackingBuffers = null,
    total_size: usize = std.math.maxInt(usize),
    err: ?anyerror = null,
};

fn calculateTotalMemorySize(buffers: *const MemoryPlannerTypes.TensorBackingBuffers) usize {
    var max_sizes = std.AutoHashMap(MemoryPlannerTypes.BufferId, usize).init(allocator);
    defer max_sizes.deinit();

    var it = buffers.iterator();
    while (it.next()) |entry| {
        const res = max_sizes.getOrPut(entry.value_ptr.id) catch continue;
        if (!res.found_existing or res.value_ptr.* < entry.value_ptr.size) {
            res.value_ptr.* = entry.value_ptr.size;
        }
    }

    var total: usize = 0;
    var size_it = max_sizes.valueIterator();
    while (size_it.next()) |size| total += size.*;
    return total;
}

pub fn codegnenerateFromGraphZant(model_name: []const u8, generated_path: []const u8, graphZant: *GraphZant) !void {
    const pre_nodes = graphZant.nodes.items.len;
    const pre_linkers = (try IR.utils.getLinkers(&IR.tensorZant_lib.tensorMap)).len;

    if (codegen_options.fuse) try graphZant.fuse(&pattern_collection.patterns);

    std.debug.print(
        "\n Nodes: {} -> {}\n Linkers: {} -> {} (Fused: {})\n",
        .{ pre_nodes, graphZant.nodes.items.len, pre_linkers, (try IR.utils.getLinkers(&IR.tensorZant_lib.tensorMap)).len, (try IR.utils.getFusedLinkers(&IR.tensorZant_lib.tensorMap)).len },
    );

    var linearizedGraph = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit(allocator);

    var backing_buffers: ?MemoryPlannerTypes.TensorBackingBuffers = null;
    defer if (backing_buffers) |*b| b.deinit();

    if (!codegen_options.dynamic and codegen_options.static_planning) {
        std.debug.assert(try graphZant.isDag(allocator) and linearizedGraph.items.len > 0);

        var results: [Planners.len]PlannerResult = undefined;
        var threads: [Planners.len]std.Thread = undefined;

        // Generic thread runner
        const Runner = struct {
            fn run(T: type, res: *PlannerResult, node: *NodeZant) void {
                var planner = T.init(allocator);
                var mp = MemoryPlannerInterface.init(&planner);
                res.buffers = mp.compute(node) catch |err| {
                    res.err = err;
                    return;
                };
                res.total_size = calculateTotalMemorySize(&res.buffers.?);
            }
        };

        // Spawn threads for each planner type
        inline for (Planners, 0..) |P, i| {
            results[i] = .{ .name = @typeName(P) };
            threads[i] = try std.Thread.spawn(.{}, Runner.run, .{ P, &results[i], linearizedGraph.items[0] });
        }

        var best_idx: usize = 0;
        std.debug.print("\n\nMemory Planner Results:", .{});

        for (threads, 0..) |t, i| {
            t.join();
            const res = &results[i];

            if (res.err) |e| {
                std.debug.print("\n  {s}: ERROR - {}", .{ res.name, e });
                if (res.buffers) |*b| b.deinit(); // Clean up partial fail
            } else {
                std.debug.print("\n  {s}: {} bytes", .{ res.name, res.total_size });
                if (res.total_size < results[best_idx].total_size) best_idx = i;
            }
        }

        const best = &results[best_idx];
        std.debug.print("\n\nBest Planner: {s} ({} bytes)\n", .{ best.name, best.total_size });

        // Keep best, free others
        for (&results, 0..) |*res, i| {
            if (i == best_idx) {
                backing_buffers = res.buffers;
            } else if (res.buffers) |*b| {
                b.deinit();
            }
        }

        // Write memory plan JSON
        if (backing_buffers) |bb| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();

            var tensors = try arena.allocator().alloc(struct {
                name: []const u8,
                size: usize,
                backing_buffer: ?MemoryPlannerTypes.BackingBuffer,
            }, bb.count());

            var it = bb.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                const t = IR.tensorZant_lib.tensorMap.get(entry.key_ptr.*).?;
                tensors[i] = .{ .name = t.name, .size = t.getSize(), .backing_buffer = entry.value_ptr.* };
            }

            const path = try std.fmt.allocPrint(allocator, "{s}memory_plan.json", .{generated_path});
            defer allocator.free(path);

            var file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            var file_buffer: [1024]u8 = undefined;
            var file_writer = file.writer(&file_buffer);
            var writer = &file_writer.interface;
            var json_str = std.json.fmt(tensors, .{});
            try json_str.format(writer);
            try writer.flush();
        }
    }

    try codegnenerateFromLinearizedGraph(model_name, generated_path, linearizedGraph, .{ .tensors_backing_buffers = backing_buffers });
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
    tensorZantMap = &IR.tensorZant_lib.tensorMap;
    try ParametersWriter.write(generated_path);
    try PredictWriter.write(generated_path, model_name, linearizedGraph, codegen_parameters);
}
