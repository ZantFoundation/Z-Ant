const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Dequant_Clip_Quant;

test "Dequant->Clip->Quant: detection succeeds on valid 3-node chain" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const clip = try h.createNode("clip1", try h.opClip());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(clip);
    defer h.destroyTestNode(quant);

    try h.wire(dequant, clip);
    try h.wire(clip, quant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, clip);
    try h.addToGraph(&graph, quant);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 3), node_list.items.len);
    try std.testing.expectEqual(dequant, node_list.items[0]);
    try std.testing.expectEqual(clip, node_list.items[1]);
    try std.testing.expectEqual(quant, node_list.items[2]);
}

test "Dequant->Clip->Quant: detection rejects non-dequant root" {
    const clip = try h.createNode("clip1", try h.opClip());
    defer h.destroyTestNode(clip);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, clip);
    try std.testing.expect(result == null);
}

test "Dequant->Clip->Quant: detection rejects when clip has wrong successor" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const clip = try h.createNode("clip1", try h.opClip());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(clip);
    defer h.destroyTestNode(relu);

    try h.wire(dequant, clip);
    try h.wire(clip, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, clip);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result == null);
}

test "Dequant->Clip->Quant: detection rejects when middle node branches" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const clip = try h.createNode("clip1", try h.opClip());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    const extra = try h.createNode("extra", try h.opRelu());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(clip);
    defer h.destroyTestNode(quant);
    defer h.destroyTestNode(extra);

    try h.wire(dequant, clip);
    try h.wire(clip, quant);
    try h.wire(clip, extra); // branching

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, clip);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result == null);
}
