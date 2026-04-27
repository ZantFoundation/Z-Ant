// Inspired by
// (https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/static-memory-planning)
const std = @import("std");
const IR = @import("IR_zant");

const GraphZant = IR.GraphZant;
const NodeZant = IR.NodeZant;
const TensorZant = IR.TensorZant;
const TensorType = IR.tensorZant_lib.TensorType;

pub const BufferId = u32;
pub const BackingBuffer = struct {
    /// A globally unique identifier among all backing buffers
    id: BufferId,
    /// Number of elements to allocate of the given elemen_type
    size: usize,
    element_type: IR.tensorZant_lib.TensorType,
    /// If t is a discrete time variable that increase by 1 for each time an
    /// operator is computed, this indicates the number of steps before this
    /// buffer is to be used
    start_borrow: usize,
    /// If t is a discrete time variable that increase by 1 for each time an
    /// operator is computed, this indicates the number of steps after which
    /// this buffer should not be used
    end_borrow: usize,
};

// Tensor name => backing buffer
pub const TensorsBackingBuffers = std.StringHashMap(BackingBuffer);

pub const TensorInfo = struct {
    name: []const u8,
    size: usize,
    ty: TensorType,
    first_step: usize,
    last_step: usize,
    liveness: usize,
};

pub const ReservedInterval = struct {
    first_step: usize,
    last_step: usize,
};

pub const PlannedBuffer = struct {
    id: BufferId,
    size: usize,
    ty: TensorType,
    reserved: std.ArrayListUnmanaged(ReservedInterval),
};

pub const BuffersByType = std.AutoHashMap(TensorType, std.ArrayListUnmanaged(PlannedBuffer));

const Borrows = std.ArrayListUnmanaged(struct {
    buffer_id: BufferId,
    tensor: *TensorZant,
});

// Intrusive node wrapper for the work queue
const CollectionType = struct {
    node: std.DoublyLinkedList.Node,
    data: *NodeZant,
};

/// Compute an associative collection that, given the name of a tensor, returns
/// a corresponding BackingBuffer that can be safely used to hold the data of
/// that tensor for the duration indicated in the BackingBuffer
pub fn computeBackingBuffers_v0(starting_node: *NodeZant, alloc: std.mem.Allocator) !TensorsBackingBuffers {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var epochs = try computeNodeEpochsFromStartNode(starting_node, arena_alloc);

    const Node = struct {
        zant_node: *NodeZant,
        epoch: usize,
    };
    var nodes_by_epoch = std.PriorityQueue(Node, void, struct {
        fn compare(context: void, a: Node, b: Node) std.math.Order {
            _ = context;
            return std.math.order(a.epoch, b.epoch);
        }
    }.compare).init(arena_alloc, undefined);

    var entry_it = epochs.iterator();
    while (entry_it.next()) |entry| {
        try nodes_by_epoch.add(.{
            .zant_node = entry.key_ptr.*,
            .epoch = entry.value_ptr.*,
        });
    }

    var free_buffers = std.AutoHashMap(BufferId, BackingBuffer).init(arena_alloc);
    var tensors_backing_buffers = TensorsBackingBuffers.init(alloc);
    // Shared as in, non-exclusive (more than one node may be reading from the
    // same tensor)
    var shared_borrows = std.AutoHashMap(*NodeZant, Borrows).init(arena_alloc);
    var backing_buffers_ref_counts = std.AutoHashMap(BufferId, usize).init(arena_alloc);
    var next_buffer_id: BufferId = 0;

    // Loop invariant: the input tensors already have a buffer assigned to them
    while (nodes_by_epoch.removeOrNull()) |node| {
        const zant_node = node.zant_node;
        const epoch = node.epoch;

        for (try zant_node.get_output_tensors()) |tensor| {
            var free_buffers_it = free_buffers.iterator();
            const duped_tensor_name = try tensors_backing_buffers.allocator.dupe(u8, tensor.name);
            var buffer_id: BufferId = undefined;
            while (free_buffers_it.next()) |entry| {
                var buffer = entry.value_ptr.*;
                // First-fit
                if (tensor.ty == buffer.element_type and buffer.size >= tensor.getSize()) {
                    buffer.start_borrow = epoch;
                    // This is the final output node, the borrow ends in one step exactly
                    if (nodes_by_epoch.count() == 0) buffer.end_borrow = epoch + 1;
                    try tensors_backing_buffers.put(duped_tensor_name, buffer);
                    _ = free_buffers.remove(buffer.id);
                    buffer_id = buffer.id;
                    break;
                }
            } else {
                // No free buffers available for the current tensor, let's make a new one
                defer next_buffer_id += 1;
                const new_buffer = BackingBuffer{
                    .size = tensor.getSize(),
                    .id = next_buffer_id,
                    .element_type = tensor.ty,
                    .start_borrow = epoch,
                    .end_borrow = if (nodes_by_epoch.count() == 0) epoch + 1 else 0,
                };
                try tensors_backing_buffers.put(duped_tensor_name, new_buffer);
                buffer_id = next_buffer_id;
            }

            try letChildrenBorrowBufferForTensor(
                zant_node,
                buffer_id,
                tensor,
                &shared_borrows,
                &backing_buffers_ref_counts,
                arena_alloc,
            );
        }

        // This node is done executing, release the borrows of this node
        if (shared_borrows.fetchRemove(zant_node)) |borrows_kv| {
            var borrows = borrows_kv.value;
            defer borrows.deinit(arena_alloc);
            while (borrows.pop()) |borrow| {
                const ref_count = backing_buffers_ref_counts.getPtr(borrow.buffer_id).?;
                var buffer = tensors_backing_buffers.getPtr(borrow.tensor.name).?;
                if (epoch > buffer.end_borrow) buffer.end_borrow = epoch;
                ref_count.* -= 1;
                if (ref_count.* == 0) {
                    var free_buffer = buffer.*;
                    free_buffer.end_borrow = 0;
                    try free_buffers.put(borrow.buffer_id, free_buffer);
                    _ = backing_buffers_ref_counts.remove(borrow.buffer_id);
                }
            }
        }
    }

    std.debug.print("\nComputed backing buffers with v0: {}", .{next_buffer_id});

    return tensors_backing_buffers;
}

/// Compute backing buffers by planning tensor lifetimes ahead of allocation.
/// v1 first records the production and last-use step for each tensor in the
/// linearized graph. It then allocates the tensors after sorting them through
/// a custom ordering **tensorInfoLessThan**, placing each tensor into the
/// smallest same-typed buffer whose reserved intervals do not overlap.
/// If no such buffer exists, a new backing buffer is created.
pub fn computeBackingBuffers_v1(
    linearized_graph: std.ArrayList(*NodeZant),
    alloc: std.mem.Allocator,
) !TensorsBackingBuffers {
    std.debug.assert(linearized_graph.items.len > 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const tensor_infos = try collectTensorInfos(linearized_graph, arena_alloc);
    std.sort.block(TensorInfo, tensor_infos.items, {}, tensorInfoLessThan);

    var buffers_by_type = BuffersByType.init(arena_alloc);
    var tensors_backing_buffers = TensorsBackingBuffers.init(alloc);
    var next_buffer_id: BufferId = 0;

    for (tensor_infos.items) |tensor| {
        var planned_buffers = try getOrCreateBuffersForType(&buffers_by_type, tensor.ty);

        const buffer_index = findBestPlannedBufferIndex(planned_buffers, tensor) orelse blk: {
            try planned_buffers.append(arena_alloc, .{
                .id = next_buffer_id,
                .size = tensor.size,
                .ty = tensor.ty,
                .reserved = .empty,
            });
            defer next_buffer_id += 1;
            break :blk planned_buffers.items.len - 1;
        };

        const planned_buffer = &planned_buffers.items[buffer_index];
        try reserveTensorInterval(planned_buffer, tensor, arena_alloc);

        const duped_tensor_name = try tensors_backing_buffers.allocator.dupe(u8, tensor.name);
        try tensors_backing_buffers.put(duped_tensor_name, .{
            .id = planned_buffer.id,
            .size = planned_buffer.size,
            .element_type = planned_buffer.ty,
            .start_borrow = tensor.first_step,
            .end_borrow = tensor.last_step,
        });
    }

    std.debug.print("\nComputed backing buffers with v1: {}", .{next_buffer_id});

    return tensors_backing_buffers;
}

// ####################################################
// HELPER FUNCTIONS FOR STATIC MEMORY PLANNING
// ####################################################

// The children of node <node> are borrowing buffer <buffer_id> (as input)
// which is holding the data for tensor <tensor>
fn letChildrenBorrowBufferForTensor(
    node: *NodeZant,
    buffer_id: BufferId,
    tensor: *TensorZant,
    shared_borrows: *std.AutoHashMap(*NodeZant, Borrows),
    ref_counts: *std.AutoHashMap(BufferId, usize),
    alloc: std.mem.Allocator,
) !void {
    // No children to borrow the lend the buffer to
    if (node.next.items.len == 0) return;
    var references: usize = 0;
    // NOTE: It is assumed that every child of the current Zant
    // node will read from the output tensor
    // This can be relaxed to further reduce peak memory usage,
    // but requires more bookkeeping and adjustments to the
    // logic
    for (node.next.items) |next_node| {
        var borrows = try shared_borrows.getOrPut(next_node);
        if (!borrows.found_existing) {
            borrows.value_ptr.* = try Borrows.initCapacity(alloc, 1);
        }
        references += 1;
        try borrows.value_ptr.append(alloc, .{
            .buffer_id = buffer_id,
            .tensor = tensor,
        });
    }

    try ref_counts.put(buffer_id, references);
}

fn computeNodeEpochsFromStartNode(
    starting_node: *NodeZant,
    alloc: std.mem.Allocator,
) !std.AutoArrayHashMap(*NodeZant, usize) {
    var epochs = std.AutoArrayHashMap(*NodeZant, usize).init(alloc);
    try epochs.put(starting_node, 1);

    var nodes: std.DoublyLinkedList = .{};
    var first_item = try alloc.create(CollectionType);
    first_item.* = .{
        .node = .{ .next = null, .prev = null },
        .data = starting_node,
    };
    nodes.append(&first_item.node);

    // First pass: compute the epoch of each node
    while (nodes.popFirst()) |node| {
        const item = @as(*CollectionType, @fieldParentPtr("node", node));
        defer alloc.destroy(item);
        const node_zant = item.data;

        const epoch = epochs.get(node_zant).?;

        for (node_zant.next.items) |next_node_zant| {
            var next_item = try alloc.create(CollectionType);
            next_item.* = .{
                .node = .{ .next = null, .prev = null },
                .data = next_node_zant,
            };

            // A node may be visited more than once (e.g. two nodes pointing to
            // the same node)
            // If we don't have an epoch yet for that node, it's the first time
            // visiting, and we give it the epoch of the parent + 1
            // If we already visited that node with a parent with an earlier
            // epoch, we update to a later one (epoch of a node = max(epochs of
            // the parents) + 1)
            const new_epoch = try epochs.getOrPut(next_node_zant);
            if (!new_epoch.found_existing or new_epoch.value_ptr.* < epoch + 1) {
                new_epoch.value_ptr.* = epoch + 1;
            }

            // Nodes may be added multiple times to the list (e.g. joins), but
            // only a finite number of times (i.e. no infinite loop)
            nodes.append(&next_item.node);
        }
    }

    return epochs;
}

fn collectTensorInfos(
    linearized_graph: std.ArrayList(*NodeZant),
    alloc: std.mem.Allocator,
) !std.ArrayListUnmanaged(TensorInfo) {
    var tensor_infos: std.ArrayListUnmanaged(TensorInfo) = .{};

    for (linearized_graph.items, 0..) |node, step| {
        for (try node.get_output_tensors()) |tensor| {
            const last_step = try computeLastStepForTensor(linearized_graph, step, tensor);

            try tensor_infos.append(alloc, .{
                .name = tensor.name,
                .size = tensor.getSize(),
                .ty = tensor.ty,
                .first_step = step,
                .last_step = last_step,
                .liveness = last_step - step,
            });
        }
    }

    return tensor_infos;
}

// fn buildNodeSteps(
//     linearized_graph: std.ArrayList(*NodeZant),
//     alloc: std.mem.Allocator,
// ) !std.AutoHashMap(*NodeZant, usize) {
//     var node_steps = std.AutoHashMap(*NodeZant, usize).init(alloc);
//     for (linearized_graph.items, 0..) |node, step| {
//         try node_steps.put(node, step);
//     }
//     return node_steps;
// }

fn nodeUsesTensor(node: *NodeZant, tensor_name: []const u8) !bool {
    for (try node.get_input_tensors()) |input_tensor| {
        if (std.mem.eql(u8, input_tensor.name, tensor_name)) {
            return true;
        }
    }
    return false;
}

fn computeLastStepForTensor(
    linearized_graph: std.ArrayList(*NodeZant),
    producer_step: usize,
    tensor: *TensorZant,
) !usize {
    var last_step: ?usize = null;

    for (linearized_graph.items[(producer_step + 1)..], producer_step + 1..) |node, step| {
        if (try nodeUsesTensor(node, tensor.name)) {
            last_step = step;
        }
    }

    if (last_step) |step| {
        return step;
    }

    // const execution_end = linearized_graph.items.len;

    // if (isGraphOutput(tensor.name, linearized_graph)) {
    //     return execution_end;
    // }

    // fallback for tensors with no internal consumers
    return producer_step + 1;
}

fn intervalsOverlapCurrentSemantics(
    a_first_step: usize,
    a_last_step: usize,
    b_first_step: usize,
    b_last_step: usize,
) bool {
    return !(a_last_step < b_first_step or b_last_step < a_first_step);
}

fn plannedBufferCanHostTensor(buffer: *const PlannedBuffer, tensor: TensorInfo) bool {
    if (buffer.ty != tensor.ty or buffer.size < tensor.size) return false;

    for (buffer.reserved.items) |reserved| {
        if (intervalsOverlapCurrentSemantics(
            reserved.first_step,
            reserved.last_step,
            tensor.first_step,
            tensor.last_step,
        )) return false;
    }

    return true;
}

fn reserveTensorInterval(
    buffer: *PlannedBuffer,
    tensor: TensorInfo,
    alloc: std.mem.Allocator,
) !void {
    try buffer.reserved.append(alloc, .{
        .first_step = tensor.first_step,
        .last_step = tensor.last_step,
    });
}

fn tensorInfoLessThan(_: void, lhs: TensorInfo, rhs: TensorInfo) bool {
    if (lhs.size * lhs.liveness != rhs.size * rhs.liveness) return lhs.size * lhs.liveness > rhs.size * rhs.liveness;
    if (lhs.size != rhs.size) return lhs.size > rhs.size;
    if (lhs.liveness != rhs.liveness) return lhs.liveness > rhs.liveness;

    // return lhs.first_step > rhs.first_step;
    return lhs.first_step < rhs.first_step;
}

fn getOrCreateBuffersForType(
    buffers_by_type: *BuffersByType,
    ty: TensorType,
) !*std.ArrayListUnmanaged(PlannedBuffer) {
    const entry = try buffers_by_type.getOrPut(ty);
    if (!entry.found_existing) entry.value_ptr.* = .empty;
    return entry.value_ptr;
}

fn findBestPlannedBufferIndex(
    buffers: *const std.ArrayListUnmanaged(PlannedBuffer),
    tensor: TensorInfo,
) ?usize {
    var best_index: ?usize = null;
    var best_size: usize = std.math.maxInt(usize);

    for (buffers.items, 0..) |*buffer, index| {
        if (!plannedBufferCanHostTensor(buffer, tensor)) continue;
        if (buffer.size < best_size) {
            best_index = index;
            best_size = buffer.size;
        }
    }

    return best_index;
}

// fn isGraphOutput(tensor_name: []const u8, linearized_graph: std.ArrayList(*NodeZant)) bool {
//     // TODO: check if there is a way to get the graph outputs
//     return false;
// }
