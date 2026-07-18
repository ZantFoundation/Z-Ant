const std = @import("std");
const IR_zant = @import("IR_zant");

const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const IR_utils = IR_zant.utils;
const TensorType = tensorZant_lib.TensorType;

/// Represents a tensor with liveness information for optimal allocation/deallocation
pub const PlanTensor = struct {
    name: []const u8,
    ty: TensorType,
    shape: []const usize,
    category: tensorZant_lib.TensorCategory,

    // Liveness information
    first_def_step: ?usize = null, // First step where tensor is defined/written
    last_use_step: ?usize = null, // Last step where tensor is used/read
    alias_of: ?[]const u8 = null, // Canonical tensor name if this tensor is an in-place alias

    /// Returns true if this tensor should be allocated at the given step
    pub fn shouldAllocateAt(self: *const PlanTensor, step: usize) bool {
        if (self.alias_of != null) return false;
        return self.first_def_step == step and
            (self.category == .LINK or self.category == .OUTPUT);
    }

    /// Returns true if this tensor should be deallocated after the given step
    pub fn shouldDeallocateAfter(self: *const PlanTensor, step: usize) bool {
        if (self.alias_of != null or self.category != .LINK) return false;
        if (self.last_use_step) |last| return last == step;
        // No consumer ever reads this tensor (e.g. dangling secondary outputs
        // like TopK's indices on a values-only graph). Free it right after the
        // step that defined it so we don't leak.
        return self.first_def_step == step;
    }
};

/// Represents a single execution step (node operation) with allocation info
pub const PlanStep = struct {
    idx: usize,
    node: *NodeZant,

    // Tensors to allocate before this step
    allocs: std.ArrayList(PlanTensor),

    // Tensors to deallocate after this step
    frees: std.ArrayList(PlanTensor),

    // Tensors that alias previously allocated buffers (in-place ops)
    aliases: std.ArrayList(PlanTensor),

    pub fn deinit(self: *PlanStep, allocator: std.mem.Allocator) void {
        self.allocs.deinit(allocator);
        self.frees.deinit(allocator);
        self.aliases.deinit(allocator);
    }
};

/// Complete execution plan with deterministic allocation/deallocation
pub const ExecutionPlan = struct {
    allocator: std.mem.Allocator,
    steps: std.ArrayList(PlanStep),
    inputs: std.ArrayList(PlanTensor),
    outputs: std.ArrayList(PlanTensor),

    pub fn init(allocator: std.mem.Allocator) ExecutionPlan {
        return ExecutionPlan{
            .allocator = allocator,
            .steps = .empty,
            .inputs = .empty,
            .outputs = .empty,
        };
    }

    pub fn deinit(self: *ExecutionPlan) void {
        for (self.steps.items) |*step| {
            step.deinit(self.allocator);
        }
        self.steps.deinit(self.allocator);
        self.inputs.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
    }
};

/// Builds execution plan from linearized graph with liveness analysis
pub fn buildExecutionPlan(allocator: std.mem.Allocator, linearizedGraph: std.ArrayList(*NodeZant)) !ExecutionPlan {
    var plan = ExecutionPlan.init(allocator);
    errdefer plan.deinit();

    // Step 1: Create tensor map with liveness info
    var tensor_liveness = std.StringHashMap(PlanTensor).init(allocator);
    defer tensor_liveness.deinit();

    try computeLiveness(&tensor_liveness, linearizedGraph);

    // Step 1.5: Detect in-place opportunities (alias tensors)
    applyInPlaceOptimizations(&tensor_liveness, linearizedGraph);

    // Step 2: Build steps with allocation info
    try buildSteps(&plan, &tensor_liveness, linearizedGraph);

    // Step 3: Skip inputs/outputs extraction for now (not critical for allocation)
    // TODO: Add proper input/output extraction when tensorZantMap is available

    return plan;
}

/// Computes liveness information for all tensors
fn computeLiveness(tensor_liveness: *std.StringHashMap(PlanTensor), linearizedGraph: std.ArrayList(*NodeZant)) !void {
    // First pass: find all tensors and their first definition
    for (linearizedGraph.items, 0..) |node, step_idx| {
        // Process output tensors (definitions)
        for (try node.get_output_tensors()) |output_tensor| {
            var plan_tensor = PlanTensor{
                .name = output_tensor.name,
                .ty = output_tensor.ty,
                .shape = output_tensor.getShape(),
                .category = output_tensor.tc,
                .first_def_step = step_idx,
                .last_use_step = null,
            };

            // Update existing or insert new
            if (tensor_liveness.get(output_tensor.name)) |existing| {
                plan_tensor.last_use_step = existing.last_use_step;
                if (existing.first_def_step) |first_def| {
                    plan_tensor.first_def_step = @min(first_def, step_idx);
                }
            }

            try tensor_liveness.put(output_tensor.name, plan_tensor);
        }
    }

    // Second pass: find last use for each tensor
    for (linearizedGraph.items, 0..) |node, step_idx| {
        // Process input tensors (uses)
        for (try node.get_input_tensors()) |input_tensor| {
            if (tensor_liveness.getPtr(input_tensor.name)) |plan_tensor| {
                plan_tensor.last_use_step = step_idx;
            }
        }
    }
}

/// Builds steps with allocation/deallocation info
fn buildSteps(plan: *ExecutionPlan, tensor_liveness: *std.StringHashMap(PlanTensor), linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items, 0..) |node, step_idx| {
        var step = PlanStep{
            .idx = step_idx,
            .node = node,
            .allocs = .empty,
            .frees = .empty,
            .aliases = .empty,
        };

        // Find tensors to allocate before this step
        var iter = tensor_liveness.iterator();
        while (iter.next()) |entry| {
            const tensor = entry.value_ptr.*;

            if (tensor.first_def_step == step_idx and tensor.alias_of != null) {
                try step.aliases.append(plan.allocator, tensor);
            }

            if (tensor.shouldAllocateAt(step_idx)) {
                try step.allocs.append(plan.allocator, tensor);
            }

            if (tensor.shouldDeallocateAfter(step_idx)) {
                try step.frees.append(plan.allocator, tensor);
            }
        }

        try plan.steps.append(plan.allocator, step);
    }
}

fn applyInPlaceOptimizations(
    tensor_liveness: *std.StringHashMap(PlanTensor),
    linearizedGraph: std.ArrayList(*NodeZant),
) void {
    for (linearizedGraph.items, 0..) |node, step_idx| {
        switch (node.op) {
            .clip => |clip_op| {
                tryAliasTensorInPlace(tensor_liveness, clip_op.input, clip_op.output, step_idx);
            },
            .fused_Dequant_Clip_Quant => |fused_op| {
                tryAliasTensorInPlace(
                    tensor_liveness,
                    fused_op.op_DequantizeLinear.x,
                    fused_op.op_QuantizeLinear.y,
                    step_idx,
                );
            },
            else => {},
        }
    }
}

fn tryAliasTensorInPlace(
    tensor_liveness: *std.StringHashMap(PlanTensor),
    input_tensor: *TensorZant,
    output_tensor: *TensorZant,
    step_idx: usize,
) void {
    if (input_tensor == output_tensor) return;

    // Only consider LINK tensors to avoid mutating graph inputs/initializers
    if (input_tensor.tc != .LINK or output_tensor.tc != .LINK) return;

    // Input and output must share type and shape
    if (input_tensor.ty != output_tensor.ty) return;
    const in_shape = input_tensor.getShape();
    const out_shape = output_tensor.getShape();
    if (in_shape.len != out_shape.len) return;
    if (!std.mem.eql(usize, in_shape, out_shape)) return;

    const input_plan = tensor_liveness.getPtr(input_tensor.name) orelse return;
    const output_plan = tensor_liveness.getPtr(output_tensor.name) orelse return;

    if (output_plan.alias_of != null) return; // Already aliased

    // Safe in-place only when this node is the final consumer of the input buffer
    if (input_plan.last_use_step == null or input_plan.last_use_step.? != step_idx) return;

    output_plan.alias_of = input_plan.name;

    // Extend the lifetime of the underlying buffer to cover the alias usage
    const new_last = output_plan.last_use_step orelse step_idx;
    extendAliasChainLastUse(tensor_liveness, input_plan.name, new_last);
}

fn extendAliasChainLastUse(
    tensor_liveness: *std.StringHashMap(PlanTensor),
    tensor_name: []const u8,
    new_last: usize,
) void {
    var current_name = tensor_name;
    while (tensor_liveness.getPtr(current_name)) |plan_tensor| {
        if (plan_tensor.last_use_step) |*last| {
            if (new_last > last.*) {
                last.* = new_last;
            }
        } else {
            plan_tensor.last_use_step = new_last;
        }

        if (plan_tensor.alias_of) |base_name| {
            current_name = base_name;
        } else {
            break;
        }
    }
}
