const IR_zant = @import("IR_zant");

const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub inline fn get_batch_normalization_output_shape(input_shape: []usize) ![]usize {
    return input_shape;
}
