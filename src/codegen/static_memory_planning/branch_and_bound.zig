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
    total_reserved: usize,
    tensor_to_buffer: std.ArrayListUnmanaged(?BufferId),
    buffers: std.ArrayListUnmanaged(PlannedBuffer),
};

/// Computes the backing buffers for the tensors in the given linearized
/// graph using a branch and bound algorithm.
pub fn computeBackingBuffers_branchAndBound(
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

    var state = BnBState{
        .buffers = .empty,
        .tensor_to_buffer = .empty,
        .total_reserved = 0,
    };
    try state.tensor_to_buffer.appendNTimes(arena_alloc, null, tensor_infos.items.len);

    var best = BnBState{
        .total_reserved = std.math.maxInt(usize),
        .tensor_to_buffer = .empty,
        .buffers = .empty,
    };

    try branchAndBoundRecursive(tensor_infos.items, 0, &state, &best, arena_alloc);

    std.debug.print("\nComputed backing buffers with branch and bound: {}", .{best.buffers.items.len});

    return buildBackingBuffersFromBestState(tensor_infos.items, &best, alloc);
}

// ###############################
// Branch and Bound helper functions
// ###############################

fn bnbLowerBound(state: *const BnBState) usize {
    return state.total_reserved;
}

/// Replaces `best` with a snapshot of the current complete search `state`.
fn bnbSnapshotBest(
    best: *BnBState,
    state: *const BnBState,
    alloc: std.mem.Allocator,
) !void {
    best.tensor_to_buffer.clearRetainingCapacity();
    try best.tensor_to_buffer.appendSlice(alloc, state.tensor_to_buffer.items);

    best.buffers.clearRetainingCapacity();
    try best.buffers.appendSlice(alloc, state.buffers.items);

    best.total_reserved = state.total_reserved;
}

/// Recursively assigns each tensor to a backing buffer using branch and bound.
/// `state` is the mutable partial solution for tensors before `tensor_index`.
/// At each level the function tries every existing compatible buffer, then the
/// option of creating a new buffer for the current tensor. After each recursive
/// call, the change is undone so the next branch starts from the same state.
/// `best` stores the smallest complete solution found so far. Branches whose
/// lower bound is already no better than `best.total_reserved` are skipped.
fn branchAndBoundRecursive(
    tensor_infos: []const TensorInfo,
    tensor_index: usize,
    state: *BnBState,
    best: *BnBState,
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

/// Given the current best state of the branch and bound and the tensor infos,
/// builds the final list of backing buffers to be returned by the main function
fn buildBackingBuffersFromBestState(
    tensor_infos: []const TensorInfo,
    best: *const BnBState,
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
