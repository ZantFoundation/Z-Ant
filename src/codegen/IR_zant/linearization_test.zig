const testing = std.testing;

const std = @import("std");
const IR_zant = @import("IR_zant");

const NodeZant = IR_zant.NodeZant;
const onnx = IR_zant.onnx;
const allocator = IR_zant.pkg_allocator.allocator;

const Tensor = IR_zant.core.tensor.Tensor;

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = .empty;
var failed_parsed_models: std.ArrayList(ErrorDetail) = .empty;

test "linearizing mnist-8 " {
    std.debug.print("\n\n ------TEST: linearizing mnist-8 ", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
    defer graphZant.deinit();

    var linearizedGraph = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit(allocator);

    std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
    for (linearizedGraph.items) |node| {
        const node_proto = node.nodeProto orelse {
            std.debug.print(" - Node: <missing node proto>\n", .{});
            continue;
        };

        std.debug.print(" - Node: {s}\n", .{node_proto.*.name orelse "<unnamed>"});
    }
}
