const std = @import("std");
const IR = @import("IR_zant");
const types = @import("../types.zig");

const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;

const BestFitDefrag = @This();

allocator: std.mem.Allocator,

// Local helper to track usage intervals for a specific buffer
const Interval = struct {
    start: usize,
    end: usize,
};

// Local tracker to speed up conflict checks without iterating the output hashmap
const BufferTracker = struct {
    allocator: std.mem.Allocator,
    id: types.BufferId,
    size: usize,
    element_type: types.ElementType,
    intervals: std.ArrayList(Interval),

    fn init(allocator: std.mem.Allocator, id: types.BufferId, size: usize, ty: types.ElementType) BufferTracker {
        return .{
            .allocator = allocator,
            .id = id,
            .size = size,
            .element_type = ty,
            .intervals = .empty,
        };
    }

    fn deinit(self: *BufferTracker) void {
        self.intervals.deinit(self.allocator);
    }

    // Check if the requested [start, end) overlaps with any existing interval
    fn hasConflict(self: *const BufferTracker, start: usize, end: usize) bool {
        for (self.intervals.items) |interval| {
            // Overlap logic: not (end <= existing.start or start >= existing.end)
            // Simplified: (start < existing.end) and (end > existing.start)
            if (start < interval.end and end > interval.start) {
                return true;
            }
        }
        return false;
    }
};

pub fn init(allocator: std.mem.Allocator) BestFitDefrag {
    return BestFitDefrag{
        .allocator = allocator,
    };
}

pub fn compute(self: *BestFitDefrag, starting_node: *NodeZant) !types.TensorBackingBuffers {
    // Phase 1: Epoch Calculation (Topological Timing)
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

    // Phase 2: Lifetime Calculation
    var tensor_lifetimes: std.ArrayList(types.TensorLifetime) = .empty;
    defer tensor_lifetimes.deinit(self.allocator);

    var entry_it = epochs.iterator();
    while (entry_it.next()) |entry| {
        const node_zant = entry.key_ptr.*;
        const epoch = entry.value_ptr.*;

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

    // Phase 3: Defragmentation Sorting
    // 1. Sort by Size (Descending) -> Fits big objects first (Tetris strategy)
    // 2. Tie-break by Start Epoch (Ascending) -> Fills time gaps linearly
    std.mem.sort(types.TensorLifetime, tensor_lifetimes.items, {}, struct {
        fn compare(context: void, a: types.TensorLifetime, b: types.TensorLifetime) bool {
            _ = context;
            if (a.size > b.size) return true;
            if (a.size < b.size) return false;
            // Secondary sort: Earlier start time preferred
            return a.start_epoch < b.start_epoch;
        }
    }.compare);

    // Phase 4: Best-Fit Allocation Loop
    var tensors_backing_buffers = types.TensorBackingBuffers.init(self.allocator);

    // We use a local list of trackers to find gaps efficiently
    var buffer_trackers: std.ArrayList(BufferTracker) = .empty;
    defer {
        for (buffer_trackers.items) |*t| t.deinit();
        buffer_trackers.deinit(self.allocator);
    }

    var next_buffer_id: types.BufferId = 0;

    for (tensor_lifetimes.items) |lifetime| {
        const tensor_name = try self.allocator.dupe(u8, lifetime.tensor.name);

        // --- Best Fit Logic ---
        // We want to find a buffer that:
        // 1. Is compatible (type)
        // 2. Is large enough (size >= needed)
        // 3. Has NO conflict in [start, end)
        // 4. Has the SMALLEST waste (buffer.size - needed)

        var best_tracker_idx: ?usize = null;
        var min_waste: usize = std.math.maxInt(usize);

        for (buffer_trackers.items, 0..) |*tracker, idx| {
            // Basic Checks
            if (tracker.element_type != lifetime.element_type) continue;
            if (tracker.size < lifetime.size) continue;

            // Conflict Check
            if (tracker.hasConflict(lifetime.start_epoch, lifetime.end_epoch)) continue;

            // Best Fit Scoring
            const waste = tracker.size - lifetime.size;

            if (waste < min_waste) {
                min_waste = waste;
                best_tracker_idx = idx;
                // Optimization: If exact fit, stop searching immediately
                if (waste == 0) break;
            }
        }

        if (best_tracker_idx) |idx| {
            // Reuse existing buffer
            var tracker = &buffer_trackers.items[idx];

            try tracker.intervals.append(self.allocator, .{ .start = lifetime.start_epoch, .end = lifetime.end_epoch });

            try tensors_backing_buffers.put(tensor_name, types.BackingBuffer{
                .size = tracker.size, // Note: We keep original buffer size, not tensor size
                .id = tracker.id,
                .element_type = tracker.element_type,
                .start_borrow = lifetime.start_epoch,
                .end_borrow = lifetime.end_epoch,
            });
        } else {
            // Create new buffer
            const new_id = next_buffer_id;
            next_buffer_id += 1;

            var new_tracker = BufferTracker.init(self.allocator, new_id, lifetime.size, lifetime.element_type);
            try new_tracker.intervals.append(self.allocator, .{ .start = lifetime.start_epoch, .end = lifetime.end_epoch });
            try buffer_trackers.append(self.allocator, new_tracker);

            try tensors_backing_buffers.put(tensor_name, types.BackingBuffer{
                .size = lifetime.size,
                .id = new_id,
                .element_type = lifetime.element_type,
                .start_borrow = lifetime.start_epoch,
                .end_borrow = lifetime.end_epoch,
            });
        }
    }

    return tensors_backing_buffers;
}

pub fn deinit(self: *BestFitDefrag) void {
    _ = self;
}

// Calculate when a tensor is last used (end of its lifetime)
fn calculateTensorEndEpoch(
    producer_node: *NodeZant,
    epochs: *std.AutoArrayHashMap(*NodeZant, usize),
) !usize {
    var max_consumer_epoch: usize = epochs.get(producer_node).?;

    for (producer_node.next.items) |consumer_node| {
        const consumer_epoch = epochs.get(consumer_node) orelse continue;
        if (consumer_epoch > max_consumer_epoch) {
            max_consumer_epoch = consumer_epoch;
        }
    }
    // Interval is [start, end)
    return max_consumer_epoch + 1;
}
