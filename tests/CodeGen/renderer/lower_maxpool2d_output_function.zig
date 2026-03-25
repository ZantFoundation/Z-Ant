const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_11 = try allocator.alloc(f32, 4);
    // SHAPE: id=0 src=0 -> shape { 1, 1, 4, 4 }
    var const_2: i32 = 0; // CONST (uop 2)
    _ = &const_2;
    var const_3: i32 = 0; // CONST (uop 3)
    _ = &const_3;
    var const_4: i32 = 2; // CONST (uop 4)
    _ = &const_4;
    var const_5: i32 = 2; // CONST (uop 5)
    _ = &const_5;
    var const_6: i32 = 1; // CONST (uop 6)
    _ = &const_6;
    var const_7: i32 = 1; // CONST (uop 7)
    _ = &const_7;
    var const_8: i32 = 4; // CONST (uop 8)
    _ = &const_8;
    var const_9: i32 = 4; // CONST (uop 9)
    _ = &const_9;
    var const_10: f32 = -std.math.inf(f32); // CONST (uop 10)
    _ = &const_10;
    var idx_12: i32 = 0; // RANGE (uop 12)
    while (idx_12 < 1) : (idx_12 += 1) {
        var idx_13: i32 = 0; // RANGE (uop 13)
        while (idx_13 < 1) : (idx_13 += 1) {
            var idx_14: i32 = 0; // RANGE (uop 14)
            while (idx_14 < 2) : (idx_14 += 1) {
                var idx_15: i32 = 0; // RANGE (uop 15)
                while (idx_15 < 2) : (idx_15 += 1) {
                    var acc_16: f32 = const_10;
                    var idx_17: i32 = 0; // RANGE (uop 17)
                    while (idx_17 < 2) : (idx_17 += 1) {
                        var idx_18: i32 = 0; // RANGE (uop 18)
                        while (idx_18 < 2) : (idx_18 += 1) {
                            const buf_19 = idx_14 * const_4; // MUL (uop 19)
                            const buf_20 = idx_17 * const_6; // MUL (uop 20)
                            const buf_21 = buf_19 + buf_20; // ADD (uop 21)
                            const buf_22 = buf_21 - const_2; // SUB (uop 22)
                            const buf_23 = idx_15 * const_5; // MUL (uop 23)
                            const buf_24 = idx_18 * const_7; // MUL (uop 24)
                            const buf_25 = buf_23 + buf_24; // ADD (uop 25)
                            const buf_26 = buf_25 - const_3; // SUB (uop 26)
                            var const_27: bool = true; // CONST (uop 27)
                            _ = &const_27;
                            var const_28: bool = false; // CONST (uop 28)
                            _ = &const_28;
                            var const_29: i32 = 0; // CONST (uop 29)
                            _ = &const_29;
                            const buf_30 = (buf_22 < const_29); // CMPLT (uop 30)
                            const buf_31 = (buf_22 < const_8); // CMPLT (uop 31)
                            const buf_32: bool = if (buf_30) const_28 else const_27; // WHERE (uop 32)
                            _ = &buf_32;
                            const buf_33: bool = if (buf_31) buf_32 else const_28; // WHERE (uop 33)
                            _ = &buf_33;
                            var const_34: i32 = 0; // CONST (uop 34)
                            _ = &const_34;
                            const buf_35 = (buf_26 < const_34); // CMPLT (uop 35)
                            const buf_36 = (buf_26 < const_9); // CMPLT (uop 36)
                            const buf_37: bool = if (buf_35) const_28 else const_27; // WHERE (uop 37)
                            _ = &buf_37;
                            const buf_38: bool = if (buf_36) buf_37 else const_28; // WHERE (uop 38)
                            _ = &buf_38;
                            const buf_39: bool = if (buf_33) buf_38 else const_28; // WHERE (uop 39)
                            _ = &buf_39;
                            if (buf_39) {
                                const addr_41 = @intFromPtr(input_0.ptr) + ((@as(usize, @intCast(idx_12)) * 16) + (@as(usize, @intCast(idx_13)) * 16) + (@as(usize, @intCast(buf_22)) * 4) + (@as(usize, @intCast(buf_26)) * 1)) * @sizeOf(f32); // GEP id=41
                                var buf_42: f32 = @as(*const f32, @ptrFromInt(addr_41)).*; // LOAD (uop 42)
                                _ = &buf_42;
                                acc_16 = @max(acc_16, buf_42); // MAX into Accumulator (uop 43)
                            }
                        }
                    }
                    const addr_47 = @intFromPtr(output_11.ptr) + ((@as(usize, @intCast(idx_12)) * 4) + (@as(usize, @intCast(idx_13)) * 4) + (@as(usize, @intCast(idx_14)) * 2) + (@as(usize, @intCast(idx_15)) * 1)) * @sizeOf(f32); // GEP id=47
                    @as(*f32, @ptrFromInt(addr_47)).* = acc_16; // STORE (uop 48)
                }
            }
        }
    }
    return output_11;
}
