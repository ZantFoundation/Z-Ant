const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_Quant_Dequant;

test "Quant->Dequant: detection succeeds on valid pattern" {
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    defer h.destroyTestNode(quant);
    defer h.destroyTestNode(dequant);

    try h.wire(quant, dequant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, quant);
    try h.addToGraph(&graph, dequant);

    const result = try Fused.fn_pattern_detection(&graph, quant);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 2), node_list.items.len);
    try std.testing.expectEqual(quant, node_list.items[0]);
    try std.testing.expectEqual(dequant, node_list.items[1]);
}

test "Quant->Dequant: detection rejects non-quant root" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    defer h.destroyTestNode(dequant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result == null);
}

test "Quant->Dequant: detection rejects quant with wrong successor" {
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(quant);
    defer h.destroyTestNode(relu);

    try h.wire(quant, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, quant);

    const result = try Fused.fn_pattern_detection(&graph, quant);
    try std.testing.expect(result == null);
}

test "Quant->Dequant: detection rejects quant with no successor" {
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    defer h.destroyTestNode(quant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, quant);

    const result = try Fused.fn_pattern_detection(&graph, quant);
    try std.testing.expect(result == null);
}
