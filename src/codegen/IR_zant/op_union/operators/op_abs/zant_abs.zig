const std = @import("std");
const IR_zant = @import("IR_zant");
const Tensor = IR_zant.core.tensor.Tensor; // Import Tensor type

pub const utils = @import("utils_abs.zig");

// --------- standard ABS
/// Compute element-wise the absolute value of the given tensor.
pub fn abs(comptime T: anytype, input: *Tensor(T)) !Tensor(T) {
    // Verify that T is among the supported types:
    // tensor(bfloat16), tensor(double), tensor(float),
    // tensor(float16), tensor(int16), tensor(int32),
    // tensor(int64), tensor(int8), tensor(uint16),
    // tensor(uint32), tensor(uint64), tensor(uint8)
    comptime if (!(std.meta.eql(T, f64) or std.meta.eql(T, f32) or std.meta.eql(T, f16) or std.meta.eql(T, f16) or std.meta.eql(T, i16) or std.meta.eql(T, i32) or std.meta.eql(T, i64) or std.meta.eql(T, i8) or std.meta.eql(T, u16) or std.meta.eql(T, u32) or std.meta.eql(T, u64) or std.meta.eql(T, u8))) {
        @compileError("Unsupported type in abs_lean");
    };

    // Allocate output tensor with the same shape as the input
    var result = try Tensor(T).fromShape(input.allocator, input.shape);

    // Perform element-wise abs computation
    try abs_lean(T, input, &result);

    return result;
}

// --------- lean ABS
pub inline fn abs_lean(comptime T: anytype, input: *Tensor(T), result: *Tensor(T)) !void {
    for (input.data, result.data) |x, *y| {
        if (std.math.isNan(x) or std.math.isInf(x)) {
            y.* = x;
        } else {
            y.* = @abs(x);
        }
    }
}
