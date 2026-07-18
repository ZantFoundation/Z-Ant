const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;
const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub inline fn lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *const Tensor(T)) !void {
    const a_rows = A.shape[A.shape.len - 2];
    const a_cols = A.shape[A.shape.len - 1];

    const b_cols = B.shape[B.shape.len - 1];

    const c_rows = a_rows;
    const c_cols = b_cols;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;

    for (0..c_rows) |row| {
        for (0..c_cols) |col| {
            for (0..a_cols) |idx| {
                const c_index = @as(u64, row) * @as(u64, c_cols) + @as(u64, col);
                const a_index = @as(u64, row) * @as(u64, a_cols) + @as(u64, idx);
                const b_index = @as(u64, idx) * @as(u64, c_cols) + @as(u64, col);

                // For integer types, use saturated arithmetic to prevent overflow
                if (@typeInfo(T) == .int) {
                    const a_val = A_ptr[a_index];
                    const b_val = B_ptr[b_index];
                    const c_val = C_ptr[c_index];

                    var result = @mulWithOverflow(a_val, b_val);
                    if (result[1] != 0) {
                        // Overflow in multiplication, saturate to max value
                        C_ptr[c_index] = std.math.maxInt(T);
                        continue;
                    }

                    result = @addWithOverflow(c_val, result[0]);
                    if (result[1] != 0) {
                        // Overflow in addition, saturate to max value
                        C_ptr[c_index] = std.math.maxInt(T);
                    } else {
                        C_ptr[c_index] = result[0];
                    }
                } else {
                    // For floating point, normal arithmetic
                    C_ptr[c_index] += A_ptr[a_index] * B_ptr[b_index];
                }
            }
        }
    }
}
