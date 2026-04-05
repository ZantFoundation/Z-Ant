const std = @import("std");
const zant = @import("zant");

const pkg_allocator = zant.utils.allocator.allocator;

pub inline fn get_batch_normalization_output_shape(input_shape: []usize) ![]usize {
    return input_shape;
}
