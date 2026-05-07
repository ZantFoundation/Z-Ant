const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Conv_Relu;

test "Conv->Relu: detection succeeds on valid pattern" {
    const conv = try h.createNode("conv1", try h.opConv());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(conv);
    defer h.destroyTestNode(relu);

    try h.wire(conv, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);
    try h.addToGraph(&graph, relu);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 2), node_list.items.len);
    try std.testing.expectEqual(conv, node_list.items[0]);
    try std.testing.expectEqual(relu, node_list.items[1]);
}

test "Conv->Relu: detection rejects non-conv root" {
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, relu);

    const result = try Fused.fn_pattern_detection(&graph, relu);
    try std.testing.expect(result == null);
}

test "Conv->Relu: detection rejects conv with wrong successor" {
    const conv = try h.createNode("conv1", try h.opConv());
    const clip = try h.createNode("clip1", try h.opClip());
    defer h.destroyTestNode(conv);
    defer h.destroyTestNode(clip);

    try h.wire(conv, clip);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);
    try h.addToGraph(&graph, clip);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}

test "Conv->Relu: detection rejects conv with multiple successors" {
    const conv = try h.createNode("conv1", try h.opConv());
    const relu1 = try h.createNode("relu1", try h.opRelu());
    const relu2 = try h.createNode("relu2", try h.opRelu());
    defer h.destroyTestNode(conv);
    defer h.destroyTestNode(relu1);
    defer h.destroyTestNode(relu2);

    try h.wire(conv, relu1);
    try h.wire(conv, relu2);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);
    try h.addToGraph(&graph, relu1);
    try h.addToGraph(&graph, relu2);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}

test "Conv->Relu: detection rejects isolated conv (no successors)" {
    const conv = try h.createNode("conv1", try h.opConv());
    defer h.destroyTestNode(conv);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}
