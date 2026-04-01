const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32 {
    const output_2 = try allocator.alloc(f32, 6);
    var idx_3: i32 = 0; // RANGE (uop 3)
    while (idx_3 < 2) : (idx_3 += 1) {
        var idx_4: i32 = 0; // RANGE (uop 4)
        while (idx_4 < 3) : (idx_4 += 1) {
            var acc_5: f32 = 0;
            var idx_6: i32 = 0; // RANGE (uop 6)
            while (idx_6 < 2) : (idx_6 += 1) {
                const addr_7 = @intFromPtr(input_0.ptr) + ((@as(usize, @intCast(idx_3)) * 2) + (@as(usize, @intCast(idx_6)) * 1)) * @sizeOf(f32); // GEP id=7
                const addr_8 = @intFromPtr(input_1.ptr) + ((@as(usize, @intCast(idx_6)) * 3) + (@as(usize, @intCast(idx_4)) * 1)) * @sizeOf(f32); // GEP id=8
                var buf_9: f32 = @as(*const f32, @ptrFromInt(addr_7)).*; // LOAD (uop 9)
                _ = &buf_9;
                var buf_10: f32 = @as(*const f32, @ptrFromInt(addr_8)).*; // LOAD (uop 10)
                _ = &buf_10;
                acc_5 += buf_9 * buf_10;
            }
            const addr_13 = @intFromPtr(output_2.ptr) + ((@as(usize, @intCast(idx_3)) * 3) + (@as(usize, @intCast(idx_4)) * 1)) * @sizeOf(f32); // GEP id=13
            @as(*f32, @ptrFromInt(addr_13)).* = acc_5; // STORE (uop 14)
        }
    }
    return output_2;
}
