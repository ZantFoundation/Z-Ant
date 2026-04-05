const std = @import("std");
const zant = @import("zant");


const pkg_allocator = zant.utils.allocator.allocator;

pub fn get_resize_output_shape(input_shape: []const usize, scales: ?[]const f32, sizes: ?[]const usize) ![]usize {
    if (scales == null and sizes == null) {
        return error.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return error.InvalidInput;
    }

    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != input_shape.len) {
            return error.InvalidInput;
        }
        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(input_shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != input_shape.len) {
            return error.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    return output_shape;
}
