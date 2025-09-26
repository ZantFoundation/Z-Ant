const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

const allocator = zant.utils.allocator.allocator;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorType = tensorZant_lib.TensorType;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant = IR_zant.NodeZant;
const Op_union = IR_zant.operators.Op_union;
const operators = IR_zant.operators;
const fused_ops = IR_zant.fused_operators;
const onnx = zant.onnx;

fn freeTensor(tensor: *TensorZant) void {
    if (tensor.ptr) |tensor_any| {
        tensor_any.deinit();
        allocator.destroy(tensor_any);
    }
    allocator.free(tensor.shape);
    allocator.free(tensor.stride);
}

fn unregisterTensor(name: []const u8) void {
    if (tensorZant_lib.tensorMap.fetchRemove(name)) |entry| {
        freeTensor(&entry.value);
    }
}

fn registerTensor(
    name: []const u8,
    ty: TensorType,
    category: TensorCategory,
    shape_values: []const usize,
) !*TensorZant {
    if (tensorZant_lib.tensorMap.fetchRemove(name)) |existing| {
        freeTensor(&existing.value);
    }

    const shape = try allocator.alloc(usize, shape_values.len);
    std.mem.copyForwards(usize, shape, shape_values);

    const stride = try TensorZant.computeStride(shape);

    var tensor = TensorZant{
        .name = name,
        .ty = ty,
        .tc = category,
        .ptr = null,
        .shape = shape,
        .stride = stride,
    };

    try tensorZant_lib.tensorMap.put(name, tensor);
    return tensorZant_lib.tensorMap.getPtr(name) orelse unreachable;
}

fn registerScalarTensor(name: []const u8, ty: TensorType, category: TensorCategory) !*TensorZant {
    return registerTensor(name, ty, category, &[_]usize{1});
}

test "fused dequant-clip-quant quantize init realigns output" {
    const x_name = "fused_align_x";
    const x_scale_name = "fused_align_x_scale";
    const x_zp_name = "fused_align_x_zp";
    const dequant_y_name = "fused_align_dequant_y";
    const clip_y_name = "fused_align_clip_y";
    const y_scale_name = "fused_align_y_scale";
    const y_zp_name = "fused_align_y_zp";
    const quant_y_name = "fused_align_quant_y";

    const x = try registerTensor(x_name, TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer unregisterTensor(x_name);

    const x_scale = try registerScalarTensor(x_scale_name, TensorType.f32, TensorCategory.INITIALIZER);
    defer unregisterTensor(x_scale_name);

    const x_zp = try registerScalarTensor(x_zp_name, TensorType.u8, TensorCategory.INITIALIZER);
    defer unregisterTensor(x_zp_name);

    const dequant_y = try registerTensor(dequant_y_name, TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer unregisterTensor(dequant_y_name);

    const clip_y = try registerTensor(clip_y_name, TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer unregisterTensor(clip_y_name);

    const y_scale = try registerScalarTensor(y_scale_name, TensorType.f32, TensorCategory.INITIALIZER);
    defer unregisterTensor(y_scale_name);

    const y_zp = try registerScalarTensor(y_zp_name, TensorType.u8, TensorCategory.INITIALIZER);
    defer unregisterTensor(y_zp_name);

    const quant_y = try registerTensor(quant_y_name, TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 48 });
    defer unregisterTensor(quant_y_name);

    var dequant_proto = onnx.NodeProto{
        .name = null,
        .op_type = "DequantizeLinear",
        .domain = null,
        .input = &[_][]const u8{ x_name, x_scale_name, x_zp_name },
        .output = &[_][]const u8{ dequant_y_name },
        .attribute = &[_]*onnx.AttributeProto{},
        .doc_string = null,
        .overload = null,
        .metadata_props = &[_]*onnx.StringStringEntryProto{},
    };

    var clip_proto = onnx.NodeProto{
        .name = null,
        .op_type = "Clip",
        .domain = null,
        .input = &[_][]const u8{ dequant_y_name },
        .output = &[_][]const u8{ clip_y_name },
        .attribute = &[_]*onnx.AttributeProto{},
        .doc_string = null,
        .overload = null,
        .metadata_props = &[_]*onnx.StringStringEntryProto{},
    };

    var quant_proto = onnx.NodeProto{
        .name = null,
        .op_type = "QuantizeLinear",
        .domain = null,
        .input = &[_][]const u8{ clip_y_name, y_scale_name, y_zp_name },
        .output = &[_][]const u8{ quant_y_name },
        .attribute = &[_]*onnx.AttributeProto{},
        .doc_string = null,
        .overload = null,
        .metadata_props = &[_]*onnx.StringStringEntryProto{},
    };

    const dequant = try operators.DequantizeLinear.init(&dequant_proto);
    const clip = try operators.Clip.init(&clip_proto);
    const quant = try operators.QuantizeLinear.init(&quant_proto);

    const updated_shape = quant.y.getShape();
    try std.testing.expectEqual(@as(usize, 50), updated_shape[3]);
    try std.testing.expectEqual(@as(usize, 48), updated_shape[2]);

    const expected_stride = try TensorZant.computeStride(updated_shape);
    defer allocator.free(expected_stride);
    try std.testing.expect(std.mem.eql(usize, expected_stride, quant.y.getStride()));

    const dequant_node = try allocator.create(NodeZant);
    defer {
        dequant_node.next.deinit();
        allocator.destroy(dequant_node);
    }
    dequant_node.* = NodeZant{
        .name = "dequant",
        .op_type = "DequantizeLinear",
        .op = Op_union{ .dequantizeLinear = dequant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    const clip_node = try allocator.create(NodeZant);
    defer {
        clip_node.next.deinit();
        allocator.destroy(clip_node);
    }
    clip_node.* = NodeZant{
        .name = "clip",
        .op_type = "Clip",
        .op = Op_union{ .clip = clip },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    const quant_node = try allocator.create(NodeZant);
    defer {
        quant_node.next.deinit();
        allocator.destroy(quant_node);
    }
    quant_node.* = NodeZant{
        .name = "quant",
        .op_type = "QuantizeLinear",
        .op = Op_union{ .quantizeLinear = quant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    var fusion_nodes = std.ArrayList(*NodeZant).init(allocator);
    defer fusion_nodes.deinit();

    try fusion_nodes.append(dequant_node);
    try fusion_nodes.append(clip_node);
    try fusion_nodes.append(quant_node);

    const fused = try fused_ops.Fused_Dequant_Clip_Quant.init_fused_op(fusion_nodes);
    const fused_shape = fused.get_output_shape();
    try std.testing.expectEqual(@as(usize, 50), fused_shape[3]);
    try std.testing.expectEqual(@as(usize, 48), fused_shape[2]);

    const outputs = try fused.get_output_tensors();
    defer allocator.free(outputs);
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expect(outputs[0] == quant.y);
}

test "fused dequant-clip-quant get_output_tensors realigns shape" {
    const x_name = "fused_realign_x";
    const x_scale_name = "fused_realign_x_scale";
    const x_zp_name = "fused_realign_x_zp";
    const dequant_y_name = "fused_realign_dequant_y";
    const clip_y_name = "fused_realign_clip_y";
    const y_scale_name = "fused_realign_y_scale";
    const y_zp_name = "fused_realign_y_zp";
    const quant_y_name = "fused_realign_quant_y";

    const x = try registerTensor(x_name, TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 8, 32, 40 });
    defer unregisterTensor(x_name);

    const x_scale = try registerScalarTensor(x_scale_name, TensorType.f32, TensorCategory.INITIALIZER);
    defer unregisterTensor(x_scale_name);

    const x_zp = try registerScalarTensor(x_zp_name, TensorType.u8, TensorCategory.INITIALIZER);
    defer unregisterTensor(x_zp_name);

    const dequant_y = try registerTensor(dequant_y_name, TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 8, 32, 40 });
    defer unregisterTensor(dequant_y_name);

    const clip_y = try registerTensor(clip_y_name, TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 8, 32, 40 });
    defer unregisterTensor(clip_y_name);

    const y_scale = try registerScalarTensor(y_scale_name, TensorType.f32, TensorCategory.INITIALIZER);
    defer unregisterTensor(y_scale_name);

    const y_zp = try registerScalarTensor(y_zp_name, TensorType.u8, TensorCategory.INITIALIZER);
    defer unregisterTensor(y_zp_name);

    const quant_y = try registerTensor(quant_y_name, TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 8, 32, 38 });
    defer unregisterTensor(quant_y_name);

    const dequant = operators.DequantizeLinear{
        .x = x,
        .x_scale = x_scale,
        .x_zero_point = x_zp,
        .y = dequant_y,
        .axis = 1,
        .block_size = 0,
        .output_dtype = TensorType.f32,
    };

    const clip = operators.Clip{
        .input = dequant_y,
        .min = null,
        .max = null,
        .output = clip_y,
    };

    const quant = operators.QuantizeLinear{
        .x = clip_y,
        .y_scale = y_scale,
        .y_zero_point = y_zp,
        .y = quant_y,
        .axis = 1,
        .block_size = 0,
        .output_dtype = TensorType.u8,
        .precision = 0,
        .saturate = 1,
    };

    const dequant_node = try allocator.create(NodeZant);
    defer {
        dequant_node.next.deinit();
        allocator.destroy(dequant_node);
    }
    dequant_node.* = NodeZant{
        .name = "dequant",
        .op_type = "DequantizeLinear",
        .op = Op_union{ .dequantizeLinear = dequant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    const clip_node = try allocator.create(NodeZant);
    defer {
        clip_node.next.deinit();
        allocator.destroy(clip_node);
    }
    clip_node.* = NodeZant{
        .name = "clip",
        .op_type = "Clip",
        .op = Op_union{ .clip = clip },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    const quant_node = try allocator.create(NodeZant);
    defer {
        quant_node.next.deinit();
        allocator.destroy(quant_node);
    }
    quant_node.* = NodeZant{
        .name = "quant",
        .op_type = "QuantizeLinear",
        .op = Op_union{ .quantizeLinear = quant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    var fusion_nodes = std.ArrayList(*NodeZant).init(allocator);
    defer fusion_nodes.deinit();

    try fusion_nodes.append(dequant_node);
    try fusion_nodes.append(clip_node);
    try fusion_nodes.append(quant_node);

    var fused = try fused_ops.Fused_Dequant_Clip_Quant.init_fused_op(fusion_nodes);

    allocator.free(quant_y.shape);
    quant_y.shape = try allocator.alloc(usize, 4);
    quant_y.shape[0] = 1;
    quant_y.shape[1] = 8;
    quant_y.shape[2] = 32;
    quant_y.shape[3] = 30;
    allocator.free(quant_y.stride);
    quant_y.stride = try TensorZant.computeStride(quant_y.shape);

    const outputs = try fused.get_output_tensors();
    defer allocator.free(outputs);

    const realigned = outputs[0].getShape();
    try std.testing.expectEqual(@as(usize, 40), realigned[3]);
    try std.testing.expectEqual(@as(usize, 32), realigned[2]);

    const expected_stride = try TensorZant.computeStride(realigned);
    defer allocator.free(expected_stride);
    try std.testing.expect(std.mem.eql(usize, expected_stride, outputs[0].getStride()));
}
