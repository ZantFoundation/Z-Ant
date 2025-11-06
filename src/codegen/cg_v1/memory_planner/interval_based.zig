const std = @import("std");
const IR = @import("IR_zant");
const types = @import("types.zig");

const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;

const IntervalBased = @This();

allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator) IntervalBased {
    return IntervalBased{
        .allocator = allocator,
    };
}

pub fn compute(self: *IntervalBased, starting_node: *NodeZant) !types.TensorBackingBuffers {
    // Phase 1: Calculate epochs for all nodes (same as greedy)
    var epochs = std.AutoArrayHashMap(*NodeZant, usize).init(self.allocator);
    defer epochs.deinit();

    try epochs.put(starting_node, 1);

    var nodes: std.DoublyLinkedList = .{};
    var first_item = try self.allocator.create(types.CollectionType);
    first_item.* = .{
        .node = .{ .next = null, .prev = null },
        .data = starting_node,
    };
    nodes.append(&first_item.node);

    while (nodes.popFirst()) |node| {
        const item = @as(*types.CollectionType, @fieldParentPtr("node", node));
        defer self.allocator.destroy(item);
        const node_zant = item.data;
        const epoch = epochs.get(node_zant).?;

        for (node_zant.next.items) |next_node_zant| {
            var next_item = try self.allocator.create(types.CollectionType);
            next_item.* = .{
                .node = .{ .next = null, .prev = null },
                .data = next_node_zant,
            };

            const new_epoch = try epochs.getOrPut(next_node_zant);
            if (!new_epoch.found_existing or new_epoch.value_ptr.* < epoch + 1) {
                new_epoch.value_ptr.* = epoch + 1;
            }

            nodes.append(&next_item.node);
        }
    }

    // Phase 2: Calculate tensor lifetimes
    var tensor_lifetimes: std.ArrayList(types.TensorLifetime) = .empty;
    defer tensor_lifetimes.deinit(self.allocator);

    var max_epoch: usize = 0;
    var entry_it = epochs.iterator();
    while (entry_it.next()) |entry| {
        const node_zant = entry.key_ptr.*;
        const epoch = entry.value_ptr.*;
        if (epoch > max_epoch) max_epoch = epoch;

        for (try node_zant.get_output_tensors()) |tensor| {
            const end_epoch = try calculateTensorEndEpoch(node_zant, &epochs);
            try tensor_lifetimes.append(self.allocator, .{
                .tensor = tensor,
                .producer_node = node_zant,
                .start_epoch = epoch,
                .end_epoch = end_epoch,
                .size = tensor.getSize(),
                .element_type = tensor.ty,
            });
        }
    }

    // Phase 3: Sort tensors by size (descending) for better packing
    std.mem.sort(types.TensorLifetime, tensor_lifetimes.items, {}, struct {
        fn compare(context: void, a: types.TensorLifetime, b: types.TensorLifetime) bool {
            _ = context;
            return a.size > b.size; // Larger tensors first
        }
    }.compare);

    // Phase 4: Allocate buffers using interval-based coloring
    var tensors_backing_buffers = types.TensorBackingBuffers.init(self.allocator);
    var buffers: std.ArrayList(types.BackingBuffer) = .empty;
    defer buffers.deinit(self.allocator);

    var next_buffer_id: types.BufferId = 0;

    for (tensor_lifetimes.items) |lifetime| {
        const tensor = lifetime.tensor;
        const tensor_name = try self.allocator.dupe(u8, tensor.name);

        // Try to find an existing buffer that:
        // 1. Has compatible type
        // 2. Is large enough
        // 3. Doesn't conflict with any tensor already assigned to it
        var assigned = false;

        for (buffers.items) |*buffer| {
            if (buffer.element_type != lifetime.element_type) continue;
            if (buffer.size < lifetime.size) continue;

            // Check if this buffer conflicts with the current tensor
            const conflicts = try bufferConflictsWithTensor(
                buffer.*,
                lifetime,
                &tensor_lifetimes,
                &tensors_backing_buffers,
            );

            if (!conflicts) {
                // This buffer can be reused
                try tensors_backing_buffers.put(tensor_name, types.BackingBuffer{
                    .size = buffer.size,
                    .id = buffer.id,
                    .element_type = buffer.element_type,
                    .start_borrow = lifetime.start_epoch,
                    .end_borrow = lifetime.end_epoch,
                });
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            // Need a new buffer
            const new_buffer = types.BackingBuffer{
                .size = lifetime.size,
                .id = next_buffer_id,
                .element_type = lifetime.element_type,
                .start_borrow = lifetime.start_epoch,
                .end_borrow = lifetime.end_epoch,
            };
            try buffers.append(self.allocator, new_buffer);
            try tensors_backing_buffers.put(tensor_name, new_buffer);
            next_buffer_id += 1;
        }
    }

    return tensors_backing_buffers;
}

pub fn deinit(self: *IntervalBased) void {
    _ = self;
}

// Calculate when a tensor is last used (end of its lifetime)
fn calculateTensorEndEpoch(
    producer_node: *NodeZant,
    epochs: *std.AutoArrayHashMap(*NodeZant, usize),
) !usize {
    var max_consumer_epoch: usize = epochs.get(producer_node).?;

    // Find all nodes that consume this tensor
    for (producer_node.next.items) |consumer_node| {
        const consumer_epoch = epochs.get(consumer_node) orelse continue;
        if (consumer_epoch > max_consumer_epoch) {
            max_consumer_epoch = consumer_epoch;
        }
    }

    // If no consumers, tensor dies immediately after production
    if (producer_node.next.items.len == 0) {
        return max_consumer_epoch + 1;
    }

    return max_consumer_epoch;
}

// Check if assigning a tensor to a buffer would create a conflict
fn bufferConflictsWithTensor(
    buffer: types.BackingBuffer,
    new_lifetime: types.TensorLifetime,
    all_lifetimes: *std.ArrayList(types.TensorLifetime),
    current_assignments: *types.TensorBackingBuffers,
) !bool {
    // Check all tensors already assigned to this buffer
    var it = current_assignments.iterator();
    while (it.next()) |entry| {
        const assigned_buffer = entry.value_ptr.*;
        if (assigned_buffer.id != buffer.id) continue;

        // Find the lifetime of this assigned tensor
        for (all_lifetimes.items) |lifetime| {
            if (std.mem.eql(u8, lifetime.tensor.name, entry.key_ptr.*)) {
                // Check if lifetimes overlap
                if (new_lifetime.overlaps(lifetime)) {
                    return true; // Conflict found
                }
                break;
            }
        }
    }

    return false; // No conflicts
}
