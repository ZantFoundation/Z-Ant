//! The aim of this is to detect well-defined sequences of nodes and sobsitute them with the fused version of those.
//! For Example:
//! Imagine in your graph you have the following sequence
//!     ...
//!     DeQuantLinear
//!     Pad
//!     QuantLinear
//!     Conv
//!     ...
//! it can be sobsitute with:
//!     ...
//!     qlinarConv
//!     ...
//! qlinarConv is a known operation the ONNX format, but for example if I want I can fuse the following:
//!     ...
//!     Conv
//!     Relu
//!     ...
//! into a custom operation called fused_Conv_Relu.
//!
//! RULES:
//!     - The name of the operations must be a concatenation of the string "fused" with names of the operations in the same order of the fusion, so
//!     the sequence: opA -> opB -> opC will be fused, if possible, into "fused_opA_opB_opC"
//!     - The pattern matcher will only fuse a math operation into an already existing math kernel
//!     - The names of the input and output tensors will be the same of the input and output tensors of the fused sequence
//!
//! Pattern Validation Logic:
//!
//!     - Start Node: Can have any number of incoming connections
//!     - Middle Nodes: Must have exactly 1 outgoing connection (no branching allowed)
//!     - End Node: Can have multiple outgoing connections (preserved in fusion)
//!     - This ensures that when we fuse DequantizeLinear -> Pad -> QuantizeLinear -> QLinearConv:
//!             -Each intermediate step has a clear, unambiguous next node
//!             - The final QLinearConv node can connect to multiple subsequent operations
//!             - No graph information is lost during the fusion process
//!     - The pattern matching will fail (return null) if any middle node has multiple outgoing connections, preventing incorrect fusions that would lose graph topology information.

const std = @import("std");
const allocator = std.heap.page_allocator; // Use your allocator

// --- Zant_IR ---
const IR_zant = @import("../IR_zant.zig");
const GraphZant = IR_zant.GraphZant;
const NodeZant = IR_zant.NodeZant;

const fused_operators = IR_zant.fused_operators;
const Op_union = IR_zant.Op_union;

/// Configuration for a fusion pattern
pub const PatternConfig = struct {
    pattern: []const []const u8, // Array of operation type strings
    name: []const u8, //used for more complex pattern like detect_qadd_pattern()
    fn_pattern_detection: *const fn (graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant), // detection stategy
    fn_pattern_fusion: *const fn (graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant, // fusion stategy
    fn_pattern_sobstitution: *const fn (graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void, // sobstitution stategy
};

/// Generic pattern matcher that can be extended for other patterns
pub fn fusePatterns(graph: *GraphZant, pattern_configs: []const PatternConfig) !void {
    for (pattern_configs) |config| {
        std.debug.print("\n\n ---------------------------------------------------------------- ", .{});
        std.debug.print("\n fusePatterns() with config ", .{});
        std.debug.print("\n ---------------------------------------------------------------- ", .{});

        for (config.pattern) |p| std.debug.print(" {s} -", .{p});
        try fusePatternsByConfig(graph, config);
    }
}

/// Generic pattern matcher using configuration
fn fusePatternsByConfig(graph: *GraphZant, config: PatternConfig) !void {
    var i: usize = 0;
    var len = graph.nodes.items.len;
    while (i < len) {
        const node = graph.nodes.items[i];
        if (try findAndFusePattern(graph, node, config)) {
            // Pattern was found and fused, restart search since graph was modified
            std.debug.print("\n -----------------------------  Pattern was found and fused, restart search since graph was modified ", .{});

            i = 0;
            len = graph.nodes.items.len;
            continue;
        }
        i += 1;
    }
}

/// Core function that coordinates pattern detection, fusion, and substitution
fn findAndFusePattern(graph: *GraphZant, root_node: *NodeZant, config: PatternConfig) !bool {
    std.debug.print("\n findAndFusePattern() starting from node type :{s} name:{s}", .{ root_node.op_type, root_node.name.? });

    // Step 1: Try to detect the pattern starting from this root node
    const maybe_node_list = try config.fn_pattern_detection(graph, root_node);

    if (maybe_node_list) |node_list| {
        // defer node_list.deinit();

        std.debug.print("\n pattern detected! Found {} nodes in pattern", .{node_list.items.len});

        // Log the detected pattern
        for (node_list.items, 0..) |node, idx| {
            std.debug.print("\n     Node[{}]: {s}", .{ idx, node.op_type });
        }

        // Step 2: Create the fused node
        const fused_node = try config.fn_pattern_fusion(graph, node_list);
        std.debug.print("\n     Fused node created: {s}", .{fused_node.op_type});

        // Step 3: Allocate the fused node on the heap to ensure it persists
        const fused_node_ptr = try allocator.create(NodeZant);
        fused_node_ptr.* = fused_node;

        // Step 4: Substitute the pattern with the fused node
        try config.fn_pattern_sobstitution(graph, fused_node_ptr, node_list);
        std.debug.print("\n     Pattern substitution completed", .{});

        return true; // Pattern was found and fused
    }

    return false; // No pattern found
}
