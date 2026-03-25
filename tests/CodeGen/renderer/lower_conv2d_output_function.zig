const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32 {
    const output_10 = try allocator.alloc(f32, 4);
    var const_0: i32 = 0; // CONST (uop 0)
    _ = &const_0;
    var const_1: i32 = 0; // CONST (uop 1)
    _ = &const_1;
    var const_2: i32 = 1; // CONST (uop 2)
    _ = &const_2;
    var const_3: i32 = 1; // CONST (uop 3)
    _ = &const_3;
    var const_4: i32 = 1; // CONST (uop 4)
    _ = &const_4;
    var const_5: i32 = 1; // CONST (uop 5)
    _ = &const_5;
    var const_6: i32 = 1; // CONST (uop 6)
    _ = &const_6;
    var const_7: i32 = 1; // CONST (uop 7)
    _ = &const_7;
    var idx_11: i32 = 0; // RANGE (uop 11)
    while (idx_11 < 1) : (idx_11 += 1) {
        var idx_12: i32 = 0; // RANGE (uop 12)
        while (idx_12 < 1) : (idx_12 += 1) {
            var idx_13: i32 = 0; // RANGE (uop 13)
            while (idx_13 < 1) : (idx_13 += 1) {
                var idx_14: i32 = 0; // RANGE (uop 14)
                while (idx_14 < 2) : (idx_14 += 1) {
                    var idx_15: i32 = 0; // RANGE (uop 15)
                    while (idx_15 < 2) : (idx_15 += 1) {
                        const buf_16 = idx_12 * const_7; // MUL (uop 16)
                        const buf_17 = buf_16 + idx_13; // ADD (uop 17)
                        var acc_18: f32 = 0;
                        var idx_19: i32 = 0; // RANGE (uop 19)
                        while (idx_19 < 1) : (idx_19 += 1) {
                            var idx_20: i32 = 0; // RANGE (uop 20)
                            while (idx_20 < 2) : (idx_20 += 1) {
                                var idx_21: i32 = 0; // RANGE (uop 21)
                                while (idx_21 < 2) : (idx_21 += 1) {
                                    const buf_22 = idx_12 * const_6; // MUL (uop 22)
                                    const buf_23 = buf_22 + idx_19; // ADD (uop 23)
                                    const buf_24 = idx_14 * const_2; // MUL (uop 24)
                                    const buf_25 = idx_20 * const_4; // MUL (uop 25)
                                    const buf_26 = buf_24 + buf_25; // ADD (uop 26)
                                    const buf_27 = buf_26 - const_0; // SUB (uop 27)
                                    const buf_28 = idx_15 * const_3; // MUL (uop 28)
                                    const buf_29 = idx_21 * const_5; // MUL (uop 29)
                                    const buf_30 = buf_28 + buf_29; // ADD (uop 30)
                                    const buf_31 = buf_30 - const_1; // SUB (uop 31)
                                    const addr_32 = @intFromPtr(input_0.ptr) + ((@as(usize, @intCast(idx_11)) * 9) + (@as(usize, @intCast(buf_23)) * 9) + (@as(usize, @intCast(buf_27)) * 3) + (@as(usize, @intCast(buf_31)) * 1)) * @sizeOf(f32); // GEP id=32
                                    const addr_33 = @intFromPtr(input_1.ptr) + ((@as(usize, @intCast(buf_17)) * 4) + (@as(usize, @intCast(idx_19)) * 4) + (@as(usize, @intCast(idx_20)) * 2) + (@as(usize, @intCast(idx_21)) * 1)) * @sizeOf(f32); // GEP id=33
                                    var buf_34: f32 = @as(*const f32, @ptrFromInt(addr_32)).*; // LOAD (uop 34)
                                    _ = &buf_34;
                                    var buf_35: f32 = @as(*const f32, @ptrFromInt(addr_33)).*; // LOAD (uop 35)
                                    _ = &buf_35;
                                    acc_18 += buf_34 * buf_35;
                                }
                            }
                        }
                        const addr_40 = @intFromPtr(output_10.ptr) + ((@as(usize, @intCast(idx_11)) * 4) + (@as(usize, @intCast(buf_17)) * 4) + (@as(usize, @intCast(idx_14)) * 2) + (@as(usize, @intCast(idx_15)) * 1)) * @sizeOf(f32); // GEP id=40
                        @as(*f32, @ptrFromInt(addr_40)).* = acc_18; // STORE (uop 41)
                    }
                }
            }
        }
    }
    return output_10;
}
