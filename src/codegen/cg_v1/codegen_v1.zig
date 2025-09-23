const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;
const IR_utils = IR.utils;
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

fn parseShapeOverride(shape_str: []const u8) ![]usize {
    var dims = std.ArrayList(usize).init(allocator);
    errdefer dims.deinit();

    var tokenizer = std.mem.tokenizeAny(u8, shape_str, ",xX []{}()|;\t\n\r");
    while (tokenizer.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t\r\n");
        if (trimmed.len == 0) continue;
        const value = try std.fmt.parseInt(usize, trimmed, 10);
        try dims.append(value);
    }

    return dims.toOwnedSlice();
}

fn overrideInputPlaceholderShape() !bool {
    if (codegen_options.shape.len == 0) return false;

    const override_dims = try parseShapeOverride(codegen_options.shape);
    defer allocator.free(override_dims);

    if (override_dims.len == 0) return false;

    const inputs = try IR_utils.getInputs(tensorZantMap);
    defer std.heap.page_allocator.free(inputs);

    if (inputs.len == 0) return false;

    for (inputs) |input_tensor| {
        if (tensorZantMap.getPtr(input_tensor.name)) |tensor_ptr| {
            const shape = tensor_ptr.getShape();
            const is_placeholder = shape.len == 0 or (shape.len == 1 and shape[0] == 1);
            if (!is_placeholder) continue;

            const new_shape = try std.heap.page_allocator.alloc(usize, override_dims.len);
            errdefer std.heap.page_allocator.free(new_shape);
            std.mem.copyForwards(usize, new_shape, override_dims);

            const new_stride = try TensorZant.computeStride(new_shape);
            errdefer std.heap.page_allocator.free(new_stride);

            const old_shape = tensor_ptr.shape;
            const old_stride = tensor_ptr.stride;

            tensor_ptr.shape = new_shape;
            tensor_ptr.stride = new_stride;

            if (old_shape.len > 0 and old_shape.ptr != new_shape.ptr) std.heap.page_allocator.free(old_shape);
            if (old_stride.len > 0 and old_stride.ptr != new_stride.ptr) std.heap.page_allocator.free(old_stride);

            std.log.info("Applied input shape override {any} to tensor '{s}'", .{ override_dims, tensor_ptr.name });
            return true;
        }
    }

    return false;
}

fn recomputeGraphShapes(linearizedGraph: std.ArrayList(*NodeZant)) void {
    for (linearizedGraph.items) |node| {
        _ = node.op.get_output_shape() catch |err| {
            const node_name = node.name orelse "<unnamed>";
            std.log.warn("Failed to recompute shape for node '{s}' ({s}): {}", .{ node_name, node.op_type, err });
        };
    }
}

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

    // --- fusion step ---
    if (codegen_options.fuse) try graphZant.fuse(&pattern_collection.patterns);

    // graphZant.print_before_linearizzation(); // DEBUG

    // Note: Pre-fusion graph printing disabled to avoid accessing freed nodes

    try graphZant.print_linearized();

    std.debug.print("\n Pre-Fusion nodes: {} \n Post-Fusion nodes: {}\n", .{ PreFusionNodes, graphZant.nodes.items.len });

    var linearizedGraph: std.ArrayList(*NodeZant) = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit();

    try codegnenerateFromLinearizedGraph(model_name, generated_path, linearizedGraph);
}

pub fn codegnenerateFromLinearizedGraph(model_name: []const u8, generated_path: []const u8, linearizedGraph: std.ArrayList(*NodeZant)) !void {

    //set globals
    tensorZantMap = &IR.tensorZant_lib.tensorMap;

    _ = try overrideInputPlaceholderShape();
    // Run a pass to ensure shapes are materialized when possible
    recomputeGraphShapes(linearizedGraph);

    try ParametersWriter.write(generated_path);

    try PredictWriter.write(generated_path, model_name, linearizedGraph);
}
