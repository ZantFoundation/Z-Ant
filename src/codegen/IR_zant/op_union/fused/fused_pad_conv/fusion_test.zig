const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Pad_Conv;

test "Pad->Conv: detection succeeds on valid pattern" {
    const pad = try h.createNode("pad1", try h.opPad());
    const conv = try h.createNode("conv1", try h.opConv());
    defer h.destroyTestNode(pad);
    defer h.destroyTestNode(conv);

    try h.wire(pad, conv);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, pad);
    try h.addToGraph(&graph, conv);

    const result = try Fused.fn_pattern_detection(&graph, pad);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 2), node_list.items.len);
    try std.testing.expectEqual(pad, node_list.items[0]);
    try std.testing.expectEqual(conv, node_list.items[1]);
}

test "Pad->Conv: detection rejects non-pad root" {
    const conv = try h.createNode("conv1", try h.opConv());
    defer h.destroyTestNode(conv);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, conv);
    try std.testing.expect(result == null);
}

test "Pad->Conv: detection rejects pad with non-conv successor" {
    const pad = try h.createNode("pad1", try h.opPad());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(pad);
    defer h.destroyTestNode(relu);

    try h.wire(pad, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, pad);

    const result = try Fused.fn_pattern_detection(&graph, pad);
    try std.testing.expect(result == null);
}

test "Pad->Conv: detection rejects pad with multiple successors" {
    const pad = try h.createNode("pad1", try h.opPad());
    const conv1 = try h.createNode("conv1", try h.opConv());
    const conv2 = try h.createNode("conv2", try h.opConv());
    defer h.destroyTestNode(pad);
    defer h.destroyTestNode(conv1);
    defer h.destroyTestNode(conv2);

    try h.wire(pad, conv1);
    try h.wire(pad, conv2);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, pad);

    const result = try Fused.fn_pattern_detection(&graph, pad);
    try std.testing.expect(result == null);
}
