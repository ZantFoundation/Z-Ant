const std = @import("std");
const IR = @import("IR_zant");
const utils = @import("utils.zig");

const NodeZant = IR.NodeZant;

pub const BufferId = utils.BufferId;
pub const BackingBuffer = utils.BackingBuffer;
const PlannedBuffer = utils.PlannedBuffer;
pub const TensorsBackingBuffers = utils.TensorsBackingBuffers;
pub const TensorInfo = utils.TensorInfo;

const BnBState = struct {
    buffers: std.ArrayListUnmanaged(PlannedBuffer),
    tensor_to_buffer: std.ArrayListUnmanaged(?BufferId),
    total_reserved: usize,
};

const BnBBest = struct {
    total_reserved: usize,
    tensor_to_buffer: std.ArrayListUnmanaged(?BufferId),
    buffers: std.ArrayListUnmanaged(PlannedBuffer),
};

/// Computes the backing buffers for the tensors in the given linearized
/// graph using a branch and bound algorithm.
pub fn computeBackingBuffers_branchAndBound(
    linearized_graph: std.ArrayList(*NodeZant),
    alloc: std.mem.Allocator,
) !TensorsBackingBuffers {
    std.debug.assert(linearized_graph.items.len > 0);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const tensor_infos = try utils.collectTensorInfos(linearized_graph, arena_alloc);
    std.sort.block(TensorInfo, tensor_infos.items, {}, tensorInfoBnBLessThan);

    var state = BnBState{
        .buffers = .empty,
        .tensor_to_buffer = .empty,
        .total_reserved = 0,
    };
    try state.tensor_to_buffer.appendNTimes(arena_alloc, null, tensor_infos.items.len);

    var best = BnBBest{
        .total_reserved = std.math.maxInt(usize),
        .tensor_to_buffer = .empty,
        .buffers = .empty,
    };

    try branchAndBoundRecursive(tensor_infos.items, 0, &state, &best, arena_alloc);

    std.debug.print("\nComputed backing buffers with branch and bound: {}", .{best.buffers.items.len});

    return buildBackingBuffersFromBnBBest(tensor_infos.items, &best, alloc);
}

// ###############################
// Branch and Bound helper functions
// ###############################

fn bnbLowerBound(state: *const BnBState) usize {
    return state.total_reserved;
}

fn bnbSnapshotBest(
    best: *BnBBest,
    state: *const BnBState,
    alloc: std.mem.Allocator,
) !void {
    best.tensor_to_buffer.clearRetainingCapacity();
    try best.tensor_to_buffer.appendSlice(alloc, state.tensor_to_buffer.items);

    best.buffers.clearRetainingCapacity();
    try best.buffers.appendSlice(alloc, state.buffers.items);

    best.total_reserved = state.total_reserved;
}

fn buildBackingBuffersFromBnBBest(
    tensor_infos: []const TensorInfo,
    best: *const BnBBest,
    alloc: std.mem.Allocator,
) !TensorsBackingBuffers {
    var tensors_backing_buffers = TensorsBackingBuffers.init(alloc);

    for (tensor_infos, 0..) |tensor, tensor_index| {
        if (tensor_index >= best.tensor_to_buffer.items.len) return error.BranchAndBoundNoSolution;

        const chosen_buffer_id = best.tensor_to_buffer.items[tensor_index] orelse return error.BranchAndBoundNoSolution;
        const chosen_buffer_index: usize = @intCast(chosen_buffer_id);
        if (chosen_buffer_index >= best.buffers.items.len) return error.BranchAndBoundNoSolution;
        const chosen_buffer = best.buffers.items[chosen_buffer_index];

        const duped_tensor_name = try tensors_backing_buffers.allocator.dupe(u8, tensor.name);
        try tensors_backing_buffers.put(duped_tensor_name, .{
            .id = chosen_buffer_id,
            .size = chosen_buffer.size,
            .element_type = chosen_buffer.ty,
            .start_borrow = tensor.first_step,
            .end_borrow = tensor.last_step,
        });
    }

    return tensors_backing_buffers;
}

fn tensorInfoBnBLessThan(_: void, lhs: TensorInfo, rhs: TensorInfo) bool {
    if (lhs.size != rhs.size) return lhs.size > rhs.size;
    if (lhs.liveness != rhs.liveness) return lhs.liveness > rhs.liveness;
    return lhs.first_step > rhs.first_step;
}

fn branchAndBoundRecursive(
    tensor_infos: []const TensorInfo,
    tensor_index: usize,
    state: *BnBState,
    best: *BnBBest,
    alloc: std.mem.Allocator,
) !void {
    if (tensor_index == tensor_infos.len) {
        if (state.total_reserved < best.total_reserved) {
            try bnbSnapshotBest(best, state, alloc);
        }
        return;
    }

    if (bnbLowerBound(state) >= best.total_reserved) return;

    const tensor = tensor_infos[tensor_index];

    var buffer_index: usize = 0;
    while (buffer_index < state.buffers.items.len) : (buffer_index += 1) {
        if (!utils.plannedBufferCanHostTensor(&state.buffers.items[buffer_index], tensor)) continue;

        const buffer_id = state.buffers.items[buffer_index].id;
        try utils.reserveTensorInterval(&state.buffers.items[buffer_index], tensor, alloc);
        state.tensor_to_buffer.items[tensor_index] = buffer_id;

        try branchAndBoundRecursive(tensor_infos, tensor_index + 1, state, best, alloc);

        state.tensor_to_buffer.items[tensor_index] = null;
        _ = state.buffers.items[buffer_index].reserved.pop().?;
    }

    const new_buffer_id: BufferId = @intCast(state.buffers.items.len);
    try state.buffers.append(alloc, .{
        .id = new_buffer_id,
        .size = tensor.size,
        .ty = tensor.ty,
        .reserved = .empty,
    });

    try utils.reserveTensorInterval(&state.buffers.items[state.buffers.items.len - 1], tensor, alloc);
    state.tensor_to_buffer.items[tensor_index] = new_buffer_id;
    state.total_reserved += tensor.size;

    try branchAndBoundRecursive(tensor_infos, tensor_index + 1, state, best, alloc);

    state.total_reserved -= tensor.size;
    state.tensor_to_buffer.items[tensor_index] = null;
    _ = state.buffers.pop().?;
}
