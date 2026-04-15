const testing = std.testing;

const std = @import("std");
const IR_zant = @import("IR_zant");

const NodeZant = IR_zant.NodeZant;
const onnx = IR_zant.onnx;
const allocator = IR_zant.pkg_allocator.allocator;

const Tensor = IR_zant.core.tensor.Tensor;

test "getInitializers() mnist-8 and TensorZant.getters" {
    std.debug.print("\n\n ------TEST: getInitializers() mnist-8 and TensorZant.getters ", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
    defer graphZant.deinit();

    var linearizedGraph = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit(allocator);

    var tensorMap = IR_zant.tensorZant_lib.tensorMap;

    const initializers = try IR_zant.utils.getInitializers(&tensorMap);

    for (initializers) |*init| {
        std.debug.print("\nname: {s} ", .{init.name});
        std.debug.print("\n     ty: {s} ", .{init.ty.toString()});
        std.debug.print("\n     tc: {s} ", .{init.tc.toString()});
        std.debug.print("\n     shape: {any} ", .{init.getShape()});
        std.debug.print("\n     shape from AnyTensor: {any} ", .{init.ptr.?.get_shape()});
        std.debug.print("\n     stride: {any} ", .{init.getStride()});
        std.debug.print("\n     size from AnyTensor: {any} ", .{init.ptr.?.get_size()});
        _ = init.ptr.?.get_data_bytes();
    }
}
