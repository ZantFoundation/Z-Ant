const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

pub fn identity(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try identity_lean(T, input, &output);

    return output;
}

pub fn identity_lean(comptime T: anytype, input: *const Tensor(T), output: *const Tensor(T)) !void {
    @memcpy(output.data, input.data);
}
