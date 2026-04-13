//! `GraphZant` — Zant's computation graph representation.
//!
//! A directed graph of `NodeZant` nodes built from an ONNX `GraphProto`.
//! Key operations:
//! - `build_graph`: constructs producer/consumer edges from the ONNX proto.
//! - `fuse`: applies kernel-fusion patterns via the pattern-matcher.
//! - `linearize`: topological sort (DFS post-order) → `ArrayList(*NodeZant)`.
//! - `isDag`: cycle detection for static memory planning pre-condition checks.
//! - `splitNodesByMemory`: partitions the linearised graph into edge/server portions.
const std = @import("std");
const IR_zant = @import("IR_zant");
const allocator = IR_zant.pkg_allocator.allocator;

// --- core ---
const Tensor = @import("core/tensor.zig");

// --- Zant IR ---
const NodeZant = @import("nodeZant.zig").NodeZant;
const tensorZant_lib = @import("tensorZant.zig");
const TensorZant = tensorZant_lib.TensorZant;

// --- onnx ---
const onnx = @import("onnx.zig");
const GraphProto = onnx.GraphProto;

// --- fusion
const pattern_matcher = @import("fusion/pattern_matcher.zig");
const PatternConfig = pattern_matcher.PatternConfig;

pub const GraphZant = struct {
    name: ?[]const u8,
    nodes: std.ArrayList(*NodeZant),
    graphProto: *GraphProto,

    /// Initializes a GraphZant instance.
    pub fn init(graphProto: *GraphProto) !GraphZant {
        return GraphZant{
            .name = graphProto.name.?,
            .nodes = .empty,
            .graphProto = graphProto,
        };
    }

    /// Deinitializes a GraphZant instance, freeing allocated resources.
    pub fn deinit(self: *GraphZant) void {
        std.debug.print("\n\ngraph.deinit() ------------- \n", .{});

        // Deinitialize each NodeZant in the nodes list REMOVED FOR TESTING
        for (self.nodes.items) |node| {
            // const n = if (node.name) |nn| nn else "<unnamed>"; // DEBUG
            // std.debug.print("\n  {s}.deinit()  ", .{n}); // DEBUG
            node.deinit();
            allocator.destroy(node); // Free the node
        }

        self.nodes.deinit(allocator);
    }

    /// Removes the specified nodes from the graph and cleans up their associated tensors.
    ///
    /// What this method DOES:
    /// - Safely removes nodes from the graph structure
    /// - Cleans up tensor references in the global tensor map
    /// - Frees memory allocated for the removed nodes
    /// - Maintains graph integrity by properly unlinking nodes
    /// - Provides debug output showing which nodes and tensors are being removed
    ///
    /// What this method DOES NOT do:
    /// - Does not update tensor references in remaining nodes (caller responsibility)
    /// - Does not validate that removal won't break graph connectivity
    /// - Does not check if removed tensors are still needed by other nodes
    /// - Does not handle graph input/output tensor updates
    /// - Does not recompute graph topology or perform graph optimization
    /// - Does not free the actual tensor data (only removes map references)
    ///
    /// IMPORTANT: After calling this method, the caller should:
    /// - Update any remaining nodes that referenced the removed tensors
    /// - Verify graph connectivity and validity
    /// - Update graph inputs/outputs if any were removed
    pub fn removeNodes(self: *GraphZant, nodes_to_remove: std.ArrayList(*NodeZant)) !void {
        if (nodes_to_remove.items.len == 0) return;

        // DEBUG
        std.debug.print("\n\nNodes to be removed:", .{});
        for (nodes_to_remove.items) |n| std.debug.print("\n      {s}", .{n.name orelse "<unnamed>"});
        std.debug.print("\n", .{});

        for (nodes_to_remove.items) |n| {
            var j: usize = 0;
            while (j < self.nodes.items.len) {
                if (self.nodes.items[j] == n) {
                    n.deinit();
                    allocator.destroy(n); // Free the node memory
                    _ = self.nodes.orderedRemove(j);
                    break;
                }
                j += 1;
            }
        }
    }

    // Adds a new node to the graph.
    pub fn build_graph(self: *GraphZant) !void {

        // create all the nodes
        for (self.graphProto.nodes) |nodeProto| {
            // allocate memory for the node
            const node = try allocator.create(NodeZant);
            node.* = try NodeZant.init(nodeProto);
            try self.nodes.append(allocator, node);
        }

        //hashmap for the outputs for the producers
        var output_map = std.StringHashMap(*NodeZant).init(allocator);
        defer output_map.deinit();

        //populate the output map with the nodes
        for (self.nodes.items) |node| {
            for (node.nodeProto.?.output) |output| {
                try output_map.put(output, node);
            }
        }

        // use the hashmap to find the producers of the inputs
        for (self.nodes.items) |customer| {
            for (customer.nodeProto.?.input) |input| {
                const producer = output_map.get(input);
                if (producer) |p| {
                    try p.add_next(customer);
                }
            }
        }

        // Perform shape inference after graph is built
        //try self.performShapeInference(&output_map);
    }

    pub fn get_predecessors(self: *GraphZant, root_node: *NodeZant) !std.ArrayList(*NodeZant) {
        var predecessors: std.ArrayList(*NodeZant) = .empty;
        for (self.nodes.items) |node| {
            // Check if this node points to our first_node
            for (node.next.items) |next_node| {
                if (next_node == root_node) {
                    try predecessors.append(allocator, node);
                    break;
                }
            }
        }

        return predecessors;
    }

    // kernel fusion and sobstitutions
    pub fn fuse(self: *GraphZant, pattern_configs: []const PatternConfig) !void {
        std.debug.print("\nGraphZant.fuse()...", .{});
        try pattern_matcher.fusePatterns(self, pattern_configs);
    }

    // linearize the graph
    pub fn linearize(self: *GraphZant, alloc: std.mem.Allocator) !std.ArrayList(*NodeZant) {
        var visited = std.AutoHashMap(*NodeZant, bool).init(alloc);
        var result: std.ArrayList(*NodeZant) = .empty;
        defer visited.deinit();

        for (self.nodes.items) |node| {
            try dfs(node, &visited, &result);
        }

        GraphZant.reverseArrayList(*NodeZant, &result);

        return result;
    }

    pub fn dfs(
        node: *NodeZant,
        visited: *std.AutoHashMap(*NodeZant, bool),
        result: *std.ArrayList(*NodeZant),
    ) !void {
        if (visited.get(node)) |_| return;

        try visited.put(node, true);

        for (node.next.items) |child| {
            try dfs(child, visited, result);
        }

        try result.append(allocator, node);
    }

    // TODO: unit tests for this
    pub fn isDag(
        self: *GraphZant,
        alloc: std.mem.Allocator,
    ) !bool {
        if (self.nodes.items.len == 0) {
            return true;
        }

        // We don't have an obvious starting point in the graph, so instead we
        // start a visit from all the nodes in the graph, unless we already ran
        // into them during a previous visit
        var visited_starting_nodes = std.AutoHashMap(*NodeZant, void).init(alloc);
        defer visited_starting_nodes.deinit();

        // Look for a cycle starting a visit from any node
        // If none is found, return true
        for (self.nodes.items) |node| {
            if (visited_starting_nodes.contains(node)) {
                continue;
            }

            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const arena_alloc = arena.allocator();

            // The boolean tells us if the visit ended already or is still in progress
            var visited = std.AutoHashMap(*NodeZant, bool).init(arena_alloc);
            const Snapshot = struct {
                zant_node: *NodeZant,
                next_index_to_check: usize,
            };

            var stack = try std.ArrayListUnmanaged(Snapshot).initCapacity(arena_alloc, self.nodes.items.len);

            try stack.append(arena_alloc, .{
                .zant_node = node,
                .next_index_to_check = 0,
            });

            visit_loop: while (stack.pop()) |stack_node| {
                const zant_node = stack_node.zant_node;
                const next_index_to_check = stack_node.next_index_to_check;
                const visited_state = try visited.getOrPut(zant_node);
                // If we have pushed a node for which the visit ended on the
                // stack, we have a bug in the algorithm
                std.debug.assert(
                    !visited_state.found_existing or
                        !visited_state.value_ptr.*,
                );

                // Visit started
                if (!visited_state.found_existing) {
                    visited_state.value_ptr.* = false;
                }

                const n_links = zant_node.next.items.len;

                std.debug.assert(next_index_to_check <= n_links);

                if (next_index_to_check == n_links) {
                    visited_state.value_ptr.* = true;
                    continue;
                }

                for (zant_node.next.items[next_index_to_check..], next_index_to_check..n_links) |next_node, i| {
                    const next_node_visited_state = try visited.getOrPut(next_node);
                    if (next_node_visited_state.found_existing and !next_node_visited_state.value_ptr.*) {
                        // We found a link from a node being visited and another node we are still visiting (i.e. a cycle)
                        return false;
                    }

                    // Visit for this next node has not started yet, add it to the stack along with its parent
                    if (!next_node_visited_state.found_existing) {
                        try stack.append(arena_alloc, .{
                            .zant_node = zant_node,
                            .next_index_to_check = i + 1,
                        });

                        try stack.append(arena_alloc, .{
                            .zant_node = next_node,
                            .next_index_to_check = 0,
                        });

                        continue :visit_loop;
                    }
                }

                visited_state.value_ptr.* = true;
                try visited_starting_nodes.put(zant_node, undefined);
            }
        }

        return true;
    }

    pub fn reverseArrayList(comptime T: type, list: *std.ArrayList(T)) void {
        var i: usize = 0;
        var j: usize = list.items.len - 1;
        while (i < j) {
            const temp = list.items[i];
            list.items[i] = list.items[j];
            list.items[j] = temp;

            i += 1;
            j -= 1;
        }
    }

    pub fn print_before_linearizzation(self: *GraphZant) void {
        std.debug.print("\n\nGraphZant: {s}\n", .{self.name orelse "<unnamed>"});
        for (self.nodes.items) |node| {
            node.print();
        }
    }

    pub fn print_linearized(self: *GraphZant) !void {
        var linearizedGraph: std.ArrayList(*NodeZant) = try self.linearize(allocator);
        defer linearizedGraph.deinit();
        std.debug.print("\n\n --- Linearized Graph post fusion : ", .{});
        for (linearizedGraph.items) |node| {
            const name = if (node.name) |n| n else "unnamed";
            std.debug.print("\n  {s} ", .{name});
        }
        std.debug.print("\n", .{});
    }

    pub fn print_detailed(self: *GraphZant) !void {
        std.debug.print("\n\nGraphZant: {s}\n", .{self.name orelse "<unnamed>"});
        for (self.nodes.items) |node| {
            std.debug.print("\n node: {s}", .{node.name orelse "<unnamed>"});
            std.debug.print("\n inputs:", .{});
            const inputs = try node.get_input_tensors();
            for (inputs) |i| std.debug.print("\n   {s}", .{i.name});
            std.debug.print("\n outputs:", .{});
            const outputs = try node.get_output_tensors();
            for (outputs) |i| std.debug.print("\n   {s}", .{i.name});
        }
    }

    // Splits a linearized graph of NodeZant into two partitions:
    // one that fits within `max_edge_memory` (bytes), and one for server execution.
    pub fn splitNodesByMemory(nodes: std.ArrayList(*NodeZant), max_edge_memory: usize) !struct {
        edge_nodes: std.ArrayList(*NodeZant),
        server_nodes: std.ArrayList(*NodeZant),
    } {
        var edge_nodes: std.ArrayList(*NodeZant) = .empty;
        var server_nodes: std.ArrayList(*NodeZant) = .empty;

        var cumulative_memory: usize = 0;

        var i: usize = 0;
        while (i < nodes.items.len) : (i += 1) {
            const node = nodes.items[i];
            const node_mem = try node.op.get_memory_footprint();

            if (cumulative_memory + node_mem <= max_edge_memory) {
                try edge_nodes.append(allocator, node);
                cumulative_memory += node_mem;
            } else {
                break;
            }
        }

        // append the remaining nodes to the server
        while (i < nodes.items.len) : (i += 1) {
            try server_nodes.append(allocator, nodes.items[i]);
        }

        return .{
            .edge_nodes = edge_nodes,
            .server_nodes = server_nodes,
        };
    }
};
