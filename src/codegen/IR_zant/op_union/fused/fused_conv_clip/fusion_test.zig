const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Conv_Clip;

test "Conv->Clip: detection succeeds on valid pattern" {
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
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 2), node_list.items.len);
    try std.testing.expectEqual(conv, node_list.items[0]);
    try std.testing.expectEqual(clip, node_list.items[1]);
}

test "Conv->Clip: detection rejects non-conv root" {
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, relu);
    try std.testing.expect(result == null);
}

test "Conv->Clip: detection rejects conv with relu successor" {
    const conv = try h.createNode("conv1", try h.opConv());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(conv);
    defer h.destroyTestNode(relu);

    try h.wire(conv, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}

test "Conv->Clip: detection rejects conv with branching" {
    const conv = try h.createNode("conv1", try h.opConv());
    const clip1 = try h.createNode("clip1", try h.opClip());
    const clip2 = try h.createNode("clip2", try h.opClip());
    defer h.destroyTestNode(conv);
    defer h.destroyTestNode(clip1);
    defer h.destroyTestNode(clip2);

    try h.wire(conv, clip1);
    try h.wire(conv, clip2);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, conv);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}
