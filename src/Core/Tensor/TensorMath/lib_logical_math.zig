//! Here you can find:
//! - Comparison Operations
//! - Logical Operations
//! - Check operation

const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;

/// Returns true if the Tensor is one-hot encoded
fn isOneHot(comptime T: anytype, t: *Tensor(T)) !bool {
    const elems_row = t.shape[t.shape.len - 1];
    if (elems_row == 0) {
        return TensorError.EmptyTensor;
    }
    const numb_rows = t.size / elems_row;
    if (numb_rows == 0) {
        return TensorError.ZeroSizeTensor;
    }

    for (0..numb_rows) |row| {
        var oneHotFound = false;
        for (0..t.shape[t.shape.len - 1]) |i| {
            if (t.data[row * elems_row + i] == 1 and !oneHotFound) {
                if (!oneHotFound) oneHotFound = true else return TensorError.NotOneHotEncoded;
            }
        }
    }

    return true;
}

/// Returns true only if all the values of shape and data are valid numbers
pub fn isSafe(comptime T: anytype, t: *Tensor(T)) !void {
    switch (@typeInfo(T)) {
        .Float => {
            // Loop over tensor data
            for (t.data) |*value| {
                if (std.math.isNan(value.*)) return TensorError.NanValue;
                if (!std.math.isFinite(value.*)) return TensorError.NotFiniteValue;
            }

            // Loop over tensor shape
            for (t.shape) |*value| {
                if (std.math.isNan(value.*)) return TensorError.NanValue;
            }
        },
        else => {
            // If T is not Float, skip isSafe checks
            return;
        },
    }
}

pub fn equal(comptime T: anytype, t1: *Tensor(T), t2: *Tensor(T)) bool {
    //same size
    if (t1.size != t2.size) {
        std.debug.print("\n\n ERROR:WRONG SIZE t1.size:{} t2.size:{}", .{ t1.size, t2.size });
        return false;
    }

    //same shape
    for (0..t1.shape.len) |i| {
        if (t1.shape[i] != t2.shape[i]) {
            std.debug.print("\n\n ERROR: WRONG SHAPE t1.shape[{}]:{} t2.shape[{}]:{}", .{ i, t1.shape[i], i, t2.shape[i] });
            return false;
        }
    }

    //same data
    if (!std.mem.eql(T, t1.data, t2.data)) {
        std.debug.print("\n\n ERROR: WRONG DATA", .{});
        return false;
    }

    return true;
}
