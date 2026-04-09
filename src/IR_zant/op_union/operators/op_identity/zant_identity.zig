const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

const pkg_allocator = zant.utils.allocator.allocator;

pub fn identity(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try identity_lean(T, input, &output);

    return output;
}

pub fn identity_lean(comptime T: anytype, input: *const Tensor(T), output: *const Tensor(T)) !void {
    @memcpy(output.data, input.data);
}
