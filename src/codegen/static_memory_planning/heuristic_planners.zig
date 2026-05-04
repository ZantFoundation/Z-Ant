const std = @import("std");
const IR = @import("IR_zant");
const utils = @import("utils.zig");

const NodeZant = IR.NodeZant;
const BackingBuffer = utils.BackingBuffer;
const BufferId = utils.BufferId;
const BuffersByType = utils.BuffersByType;
const Borrows = utils.Borrows;
const TensorInfo = utils.TensorInfo;
const TensorsBackingBuffers = utils.TensorsBackingBuffers;

/// Compute an associative collection that, given the name of a tensor, returns
/// a corresponding BackingBuffer that can be safely used to hold the data of
/// that tensor for the duration indicated in the BackingBuffer
/// Inspired by
/// (https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/static-memory-planning)
pub fn computeBackingBuffers_v0(starting_node: *NodeZant, alloc: std.mem.Allocator) !TensorsBackingBuffers {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var epochs = try utils.computeNodeEpochsFromStartNode(starting_node, arena_alloc);

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

            try utils.letChildrenBorrowBufferForTensor(
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
/// smallest same-typed buffer whose assigned intervals do not overlap.
/// If no such buffer exists, a new backing buffer is created.
pub fn computeBackingBuffers_v1(
    linearized_graph: std.ArrayList(*NodeZant),
    alloc: std.mem.Allocator,
    static_planning_option: []const u8,
) !TensorsBackingBuffers {
    std.debug.assert(linearized_graph.items.len > 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const tensor_infos = try utils.collectTensorInfos(linearized_graph, arena_alloc);
    std.sort.block(TensorInfo, tensor_infos.items, static_planning_option, utils.tensorInfoLessThan);

    var buffers_by_type = BuffersByType.init(arena_alloc);
    var tensors_backing_buffers = TensorsBackingBuffers.init(alloc);
    var next_buffer_id: BufferId = 0;

    for (tensor_infos.items) |tensor| {
        var planned_buffers = try utils.getOrCreateBuffersForType(&buffers_by_type, tensor.ty);

        const buffer_index = utils.findBestPlannedBufferIndex(planned_buffers, tensor) orelse blk: {
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
        try utils.reserveTensorInterval(planned_buffer, tensor, arena_alloc);

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
