const std = @import("std");
const IR = @import("IR_zant");

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

pub const StaticPlanningOptions = struct {
    pub const disabled = "disabled";
    pub const enabled = "enabled";
    pub const default_size = "default_size";
    pub const default_liveness = "default_liveness";
    pub const liveness_first = "liveness_first";
    pub const size_first = "size_first";
    pub const inverse_first_step = "_inverse_first_step";

    pub fn isValid(option: []const u8) bool {
        if (hasInverseFirstStep(option)) {
            const base_option = baseOption(option);
            if (std.mem.eql(u8, base_option, disabled) or
                std.mem.eql(u8, base_option, enabled))
            {
                return false;
            }
        }
        return isValidBase(baseOption(option));
    }

    pub fn isEnabled(option: []const u8) bool {
        return !std.mem.eql(u8, option, disabled);
    }

    pub fn hasInverseFirstStep(option: []const u8) bool {
        return std.mem.endsWith(u8, option, inverse_first_step);
    }

    pub fn baseOption(option: []const u8) []const u8 {
        if (!hasInverseFirstStep(option)) return option;
        return option[0 .. option.len - inverse_first_step.len];
    }

    fn isValidBase(option: []const u8) bool {
        return std.mem.eql(u8, option, disabled) or
            std.mem.eql(u8, option, enabled) or
            std.mem.eql(u8, option, default_size) or
            std.mem.eql(u8, option, default_liveness) or
            std.mem.eql(u8, option, liveness_first) or
            std.mem.eql(u8, option, size_first);
    }
};

pub const AssignedInterval = struct {
    first_step: usize,
    last_step: usize,
};

pub const PlannedBuffer = struct {
    id: BufferId,
    size: usize,
    ty: TensorType,
    reserved: std.ArrayListUnmanaged(AssignedInterval),
};

pub const BuffersByType = std.AutoHashMap(TensorType, std.ArrayListUnmanaged(PlannedBuffer));

pub const Borrows = std.ArrayListUnmanaged(struct {
    buffer_id: BufferId,
    tensor: *TensorZant,
});

// Intrusive node wrapper for the work queue
const CollectionType = struct {
    node: std.DoublyLinkedList.Node,
    data: *NodeZant,
};

/// This is the comparator used for tensors in heuristic v1, it can be used
/// like so, or it can be modified to better fit the model it's being used for
/// by using different build flags (see `docs/BUILD_FLAGS.md`)
/// Note: go to `docs/heuristics_for_static_memory_planning.md` for more details
/// on the heuristic and how to modify this comparator to fit your model
pub fn tensorInfoLessThan(static_planning_option: []const u8, lhs: TensorInfo, rhs: TensorInfo) bool {
    std.debug.assert(StaticPlanningOptions.isValid(static_planning_option));
    const base_option = StaticPlanningOptions.baseOption(static_planning_option);
    const inverse_first_step = StaticPlanningOptions.hasInverseFirstStep(static_planning_option);

    if (std.mem.eql(u8, base_option, StaticPlanningOptions.enabled) or
        std.mem.eql(u8, base_option, StaticPlanningOptions.default_size))
    {
        // `enabled` uses the same ordering as `default_size`.
        if (lhs.size * lhs.liveness != rhs.size * rhs.liveness) return lhs.size * lhs.liveness > rhs.size * rhs.liveness;
        if (lhs.size != rhs.size) return lhs.size > rhs.size;
    }

    if (std.mem.eql(u8, base_option, StaticPlanningOptions.default_liveness)) {
        if (lhs.size * lhs.liveness != rhs.size * rhs.liveness) return lhs.size * lhs.liveness > rhs.size * rhs.liveness;
        if (lhs.liveness != rhs.liveness) return lhs.liveness > rhs.liveness;
    }

    if (std.mem.eql(u8, base_option, StaticPlanningOptions.liveness_first)) {
        if (lhs.liveness != rhs.liveness) return lhs.liveness > rhs.liveness;
        if (lhs.size != rhs.size) return lhs.size > rhs.size;
    }

    if (std.mem.eql(u8, base_option, StaticPlanningOptions.size_first)) {
        if (lhs.size != rhs.size) return lhs.size > rhs.size;
        if (lhs.liveness != rhs.liveness) return lhs.liveness > rhs.liveness;
    }

    return if (inverse_first_step) lhs.first_step > rhs.first_step else lhs.first_step < rhs.first_step;
}

// The children of node <node> are borrowing buffer <buffer_id> (as input)
// which is holding the data for tensor <tensor>
pub fn letChildrenBorrowBufferForTensor(
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

pub fn computeNodeEpochsFromStartNode(
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

pub fn collectTensorInfos(
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

fn nodeUsesTensor(node: *NodeZant, tensor_name: []const u8) !bool {
    for (try node.get_input_tensors()) |input_tensor| {
        if (std.mem.eql(u8, input_tensor.name, tensor_name)) {
            return true;
        }
    }
    return false;
}

pub fn computeLastStepForTensor(
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

    // fallback for tensors with no internal consumers
    return producer_step + 1;
}

pub fn intervalsOverlapCurrentSemantics(
    a_first_step: usize,
    a_last_step: usize,
    b_first_step: usize,
    b_last_step: usize,
) bool {
    return !(a_last_step < b_first_step or b_last_step < a_first_step);
}

/// Returns wether a buffer can host a given tensor. It double checks the type even
/// though the heuristic already sorts the tensors by type and only compares tensors
/// to buffers of the same type
pub fn plannedBufferCanHostTensor(buffer: *const PlannedBuffer, tensor: TensorInfo) bool {
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

/// Adds tensor lifetieme intervals to the given planned buffer's reserved list
pub fn reserveTensorInterval(
    buffer: *PlannedBuffer,
    tensor: TensorInfo,
    alloc: std.mem.Allocator,
) !void {
    try buffer.reserved.append(alloc, .{
        .first_step = tensor.first_step,
        .last_step = tensor.last_step,
    });
}

/// Returns the list of 'planned buffers' that can host a given tensor type,
/// creating an entry for the type if it doesn't exist yet in the map passed as argument
pub fn getOrCreateBuffersForType(
    buffers_by_type: *BuffersByType,
    ty: TensorType,
) !*std.ArrayListUnmanaged(PlannedBuffer) {
    const entry = try buffers_by_type.getOrPut(ty);
    if (!entry.found_existing) entry.value_ptr.* = .empty;
    return entry.value_ptr;
}

pub fn findBestPlannedBufferIndex(
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

pub fn shouldUseBranchAndBound(node_count: usize) bool {
    return node_count <= 25;
}
