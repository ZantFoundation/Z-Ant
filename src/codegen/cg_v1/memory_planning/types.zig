const std = @import("std");

const IR = @import("IR_zant");

pub const BufferId = u32;

pub const BackingBuffer = struct {
    id: BufferId,
    size: usize,
    element_type: IR.tensorZant_lib.TensorType,
    start_borrow: usize,
    end_borrow: usize,
};

pub const TensorBackingBuffers = std.StringHashMap(BackingBuffer);
