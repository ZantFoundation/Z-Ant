const testing = std.testing;

const std = @import("std");
const IR_zant = @import("IR_zant");

const onnx = IR_zant.onnx;
const allocator = IR_zant.pkg_allocator.allocator;

const Tensor = IR_zant.core.tensor.Tensor;

test "parsing mnist-8 graphZant" {
    std.debug.print("\n\n ------TEST: parsing mnist-8 graphZant", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
    defer graphZant.deinit();

    //USELESS SHIT FOR DEBUG
    // std.debug.print("__HASH_MAP__", .{});
    // var it = IR_zant.tensorZant_lib.tensorMap.iterator();
    // while (it.next()) |entry| {
    //     std.debug.print("Key: {s}, Value.ty: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.name });
    // }
}
