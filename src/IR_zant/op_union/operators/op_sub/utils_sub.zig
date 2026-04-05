const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

const zant_sub = @import("zant_sub.zig");

// Helper for mixed precision subtraction
pub inline fn sub_lean_mixed(comptime T1: type, comptime T2: type, comptime outputType: anytype, t1: *Tensor(T1), t2: *Tensor(T2), outputTensor: *Tensor(outputType)) !void {
    if (T1 == T2) {
        // Same types - use the regular function
        return zant_sub.sub_lean(T1, outputType, t1, @ptrCast(t2), outputTensor);
    } else {
        // Different types - convert both to output type for computation
        const rank1 = t1.shape.len;
        const rank2 = t2.shape.len;

        // Fast path: identical shapes, no broadcasting needed
        if (rank1 == rank2 and std.mem.eql(usize, t1.shape, t2.shape)) {
            for (0..t1.size) |i| {
                const val1: outputType = if (T1 == outputType) t1.data[i] else @floatCast(t1.data[i]);
                const val2: outputType = if (T2 == outputType) t2.data[i] else @floatCast(t2.data[i]);
                outputTensor.data[i] = val1 - val2;
            }
        } else {
            // TODO: implement broadcasting for mixed types
            return error.BroadcastingNotSupportedMixed;
        }
    }
}
