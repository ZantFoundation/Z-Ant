const std = @import("std");
const h = @import("../fusion_test_utils.zig");
const IR_zant = @import("IR_zant");
const Fused = IR_zant.fused_operators.Fused_2Dequant_Add_Quant;

test "2Dequant->Add->Quant: detection succeeds on valid diamond pattern" {
    //  dequant_a -\
    //              -> add -> quant
    //  dequant_b -/
    const dequant_a = try h.createNode("dequant_a", try h.opDequantizeLinear());
    const dequant_b = try h.createNode("dequant_b", try h.opDequantizeLinear());
    const add = try h.createNode("add1", try h.opAdd());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    defer h.destroyTestNode(dequant_a);
    defer h.destroyTestNode(dequant_b);
    defer h.destroyTestNode(add);
    defer h.destroyTestNode(quant);

    try h.wire(dequant_a, add);
    try h.wire(dequant_b, add);
    try h.wire(add, quant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant_a);
    try h.addToGraph(&graph, dequant_b);
    try h.addToGraph(&graph, add);
    try h.addToGraph(&graph, quant);

    // Detection starts from the Add node
    const result = try Fused.fn_pattern_detection(&graph, add);
    try std.testing.expect(result != null);

    var node_list = result.?;
    defer node_list.deinit(h.allocator);
    try std.testing.expectEqual(@as(usize, 4), node_list.items.len);
    // items[0] and items[1] are the two dequant nodes (order depends on graph iteration)
    try std.testing.expect(node_list.items[0].op == .dequantizeLinear);
    try std.testing.expect(node_list.items[1].op == .dequantizeLinear);
    try std.testing.expectEqual(add, node_list.items[2]);
    try std.testing.expectEqual(quant, node_list.items[3]);
}

test "2Dequant->Add->Quant: detection rejects non-add root" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    defer h.destroyTestNode(dequant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);

    const result = try Fused.fn_pattern_detection(&graph, dequant);
    try std.testing.expect(result == null);
}

test "2Dequant->Add->Quant: detection rejects add with non-dequant predecessor" {
    //  relu -\
    //         -> add -> quant
    //  dequant -/
    const relu = try h.createNode("relu1", try h.opRelu());
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const add = try h.createNode("add1", try h.opAdd());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    defer h.destroyTestNode(relu);
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(add);
    defer h.destroyTestNode(quant);

    try h.wire(relu, add);
    try h.wire(dequant, add);
    try h.wire(add, quant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, relu);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, add);
    try h.addToGraph(&graph, quant);

    const result = try Fused.fn_pattern_detection(&graph, add);
    try std.testing.expect(result == null);
}

test "2Dequant->Add->Quant: detection rejects add with wrong successor" {
    const dequant_a = try h.createNode("dequant_a", try h.opDequantizeLinear());
    const dequant_b = try h.createNode("dequant_b", try h.opDequantizeLinear());
    const add = try h.createNode("add1", try h.opAdd());
    const relu = try h.createNode("relu1", try h.opRelu());
    defer h.destroyTestNode(dequant_a);
    defer h.destroyTestNode(dequant_b);
    defer h.destroyTestNode(add);
    defer h.destroyTestNode(relu);

    try h.wire(dequant_a, add);
    try h.wire(dequant_b, add);
    try h.wire(add, relu);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant_a);
    try h.addToGraph(&graph, dequant_b);
    try h.addToGraph(&graph, add);
    try h.addToGraph(&graph, relu);

    const result = try Fused.fn_pattern_detection(&graph, add);
    try std.testing.expect(result == null);
}

test "2Dequant->Add->Quant: detection rejects add with single predecessor" {
    const dequant = try h.createNode("dequant1", try h.opDequantizeLinear());
    const add = try h.createNode("add1", try h.opAdd());
    const quant = try h.createNode("quant1", try h.opQuantizeLinear());
    defer h.destroyTestNode(dequant);
    defer h.destroyTestNode(add);
    defer h.destroyTestNode(quant);

    try h.wire(dequant, add);
    try h.wire(add, quant);

    var graph = h.createGraph();
    defer graph.nodes.deinit(h.allocator);
    try h.addToGraph(&graph, dequant);
    try h.addToGraph(&graph, add);
    try h.addToGraph(&graph, quant);

    const result = try Fused.fn_pattern_detection(&graph, add);
    try std.testing.expect(result == null);
}
