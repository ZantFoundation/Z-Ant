const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_2 = try allocator.alloc(f32, 24);
    // SHAPE: id=0 src=0 -> shape { 2, 3, 4 }
    var idx_4: i32 = 0; // RANGE (uop 4)
    while (idx_4 < 2) : (idx_4 += 1) {
        var idx_5: i32 = 0; // RANGE (uop 5)
        while (idx_5 < 3) : (idx_5 += 1) {
            var idx_6: i32 = 0; // RANGE (uop 6)
            while (idx_6 < 4) : (idx_6 += 1) {
                const addr_7 = @intFromPtr(input_0.ptr) + ((@as(usize, @intCast(idx_4)) * 12) + (@as(usize, @intCast(idx_5)) * 4) + (@as(usize, @intCast(idx_6)) * 1)) * @sizeOf(f32); // GEP id=7
                var buf_8: f32 = @as(*const f32, @ptrFromInt(addr_7)).*; // LOAD (uop 8)
                _ = &buf_8;
                const addr_9 = @intFromPtr(output_2.ptr) + ((@as(usize, @intCast(idx_4)) * 12) + (@as(usize, @intCast(idx_5)) * 4) + (@as(usize, @intCast(idx_6)) * 1)) * @sizeOf(f32); // GEP id=9
                @as(*f32, @ptrFromInt(addr_9)).* = buf_8; // STORE (uop 10)
            }
        }
    }
    return output_2;
}
