const std = @import("std");

const IR = @import("IR_zant");

pub const BufferId = u32;

pub const ElementType = IR.tensorZant_lib.TensorType;

pub const BackingBuffer = struct {
    id: BufferId,
    size: usize,
    element_type: ElementType,
    start_borrow: usize,
    end_borrow: usize,
};

pub const TensorLifetime = struct {
    tensor: *IR.TensorZant,
    producer_node: *IR.NodeZant,
    start_epoch: usize,
    end_epoch: usize,
    size: usize,
    element_type: ElementType,

    pub fn overlaps(self: TensorLifetime, other: TensorLifetime) bool {
        return !(self.end_epoch <= other.start_epoch or other.end_epoch <= self.start_epoch);
    }
};

pub const CollectionType = struct {
    node: std.DoublyLinkedList.Node,
    data: *IR.NodeZant,
};

pub const TensorBackingBuffers = std.StringHashMap(BackingBuffer);
