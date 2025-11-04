const std = @import("std");
const types = @import("types.zig");
const IR = @import("IR_zant");

const NodeZant = IR.NodeZant;
const TensorZant = IR.TensorZant;

const MemoryPlanner = @This();

ptr: *anyopaque,
computeOpaquePtr: *const fn (ptr: *anyopaque, starting_node: *NodeZant) anyerror!types.TensorBackingBuffers,
deinitOpaquePtr: *const fn (ptr: *anyopaque) void,

pub fn init(memory_planner_ptr: anytype) MemoryPlanner {
    const T = @TypeOf(memory_planner_ptr);

    const gen = struct {
        pub fn computeOpaque(ptr: *anyopaque, starting_node: *NodeZant) anyerror!types.TensorBackingBuffers {
            const self: T = @ptrCast(@alignCast(ptr));
            return try self.compute(starting_node);
        }

        pub fn deinit(ptr: *anyopaque) void {
            const self: T = @ptrCast(@alignCast(ptr));
            self.deinit();
        }
    };

    return MemoryPlanner{
        .ptr = memory_planner_ptr,
        .computeOpaquePtr = gen.computeOpaque,
        .deinitOpaquePtr = gen.deinit,
    };
}

pub fn compute(self: *MemoryPlanner, starting_node: *NodeZant) anyerror!types.TensorBackingBuffers {
    return try self.computeOpaquePtr(self.ptr, starting_node);
}

pub fn deinit(self: *MemoryPlanner) void {
    self.deinitOpaquePtr(self.ptr);
}
