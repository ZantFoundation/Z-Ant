const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = IR_zant.pkg_allocator.allocator;

/// The Sigmoid activation function is a smooth, S-shaped function that maps any input
/// to a value between 0 and 1.
/// it can suffer from vanishing gradients, especially for large positive or negative
/// inputs, slowing down training in deep networks.
pub inline fn sigmoid(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return error.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try sigmoid_lean(T, tensor, &output_tensor);

    return output_tensor;
}

pub inline fn sigmoid_lean(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    @setEvalBranchQuota(100000);
    //std.log.debug("\n[DEBUG] sigmoid_lean:", .{});
    //std.log.debug("\n  Input shape: ", .{});
    //for (input_tensor.shape) |s| std.log.debug("{d} ", .{s});

    //std.log.debug("\n  Output shape: ", .{});
    //for (output_tensor.shape) |s| std.log.debug("{d} ", .{s});

    //apply Sigmoid
    for (0..input_tensor.size) |i| {
        const input_val = input_tensor.data[i];
        output_tensor.data[i] = 1.0 / (1.0 + @exp(-input_val));
        //std.log.debug("\n  sigmoid({d:.6}) = {d:.6}", .{ input_val, output_tensor.data[i] });
    }
    //std.log.debug("\n[DEBUG] sigmoid_lean completed\n", .{});
}
