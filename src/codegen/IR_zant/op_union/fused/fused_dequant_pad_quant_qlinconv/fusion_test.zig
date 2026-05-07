const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Dequant_Pad_Quant_QLinConv;

test "Dequant->Pad->Quant->QLinConv: detection succeeds on valid 4-node chain" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const pad = try h.createNode("pad1", try h.opPad());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    const qlinconv = try h.createNode("qlinconv1", try h.opQLinearConv());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(pad);
    defer h.destroyTestNode(quant);
    defer h.destroyTestNode(qlinconv);

    try h.wire(dequant, pad);
    try h.wire(pad, quant);
    try h.wire(quant, qlinconv);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, pad);
    try h.addToGraph(&graph, quant);
    try h.addToGraph(&graph, qlinconv);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 4), node_list.items.len);
    try std.testing.expectEqual(dequant, node_list.items[0]);
    try std.testing.expectEqual(pad, node_list.items[1]);
    try std.testing.expectEqual(quant, node_list.items[2]);
    try std.testing.expectEqual(qlinconv, node_list.items[3]);
}

test "Dequant->Pad->Quant->QLinConv: detection rejects broken chain (wrong third node)" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const pad = try h.createNode("pad1", try h.opPad());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(pad);
    defer h.destroyTestNode(relu);

    try h.wire(dequant, pad);
    try h.wire(pad, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, pad);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result == null);
}

test "Dequant->Pad->Quant->QLinConv: detection rejects non-dequant root" {
    const pad = try h.createNode("pad1", try h.opPad());
    defer h.destroyTestNode(pad);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, pad);
    try std.testing.expect(result == null);
}
