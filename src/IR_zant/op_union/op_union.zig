const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
pub const operators = @import("operators/operators.zig");
pub const fused_operators = @import("fused_operators/fused_operators.zig");

const tensorZant = @import("../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const nodeZant = @import("../nodeZant.zig");
const NodeZant = nodeZant.NodeZant;

// --- uops ---
const UOpBuilder = zant.uops.UOpBuilder;

pub const Op_union = union(enum) {
    // ------------- atomic operations
    add: operators.Add,
    averagePool: operators.AveragePool,
    batchNormalization: operators.BatchNormalization,
    cast: operators.Cast,
    ceil: operators.Ceil,
    clip: operators.Clip,
    concat: operators.Concat,
    constant: operators.Constant,
    conv: operators.Conv,
    convInteger: operators.ConvInteger,
    dequantizeLinear: operators.DequantizeLinear,
    div: operators.Div,
    dynamicQuantizeLinear: operators.DynamicQuantizeLinear,
    elu: operators.Elu,
    exp: operators.Exp,
    flatten: operators.Flatten,
    floor: operators.Floor,
    gather: operators.Gather,
    gatherND: operators.GatherND,
    gemm: operators.Gemm,
    gelu: operators.Gelu,
    globalAveragePool: operators.GlobalAveragePool,
    identity: operators.Identity,
    leakyRelu: operators.LeakyRelu,
    matMul: operators.MatMul,
    maxPool: operators.MaxPool,
    min: operators.Min,
    mul: operators.Mul,
    neg: operators.Neg,
    nonMaxSuppression: operators.NonMaxSuppression,
    oneHot: operators.OneHot,
    pad: operators.Pad,
    pow: operators.Pow,
    qgemm: operators.QGemm,
    qlinearadd: operators.QLinearAdd,
    qlinearaveragepool: operators.QLinearAveragePool,
    qlinearconcat: operators.QLinearConcat,
    qlinearconv: operators.QLinearConv,
    qlinearglobalaveragepool: operators.QLinearGlobalAveragePool,
    qlinearmatmul: operators.QLinearMatMul,
    qlinearmul: operators.QLinearMul,
    qlinearsoftmax: operators.QLinearSoftmax,
    quantizeLinear: operators.QuantizeLinear,
    reduceMean: operators.ReduceMean,
    relu: operators.Relu,
    reshape: operators.Reshape,
    resize: operators.Resize,
    shape: operators.Shape,
    sigmoid: operators.Sigmoid,
    slice: operators.Slice,
    softmax: operators.Softmax,
    split: operators.Split,
    squeeze: operators.Squeeze,
    sqrt: operators.Sqrt,
    sub: operators.Sub,
    topK: operators.TopK,
    tanh: operators.Tanh,
    transpose: operators.Transpose,
    unsqueeze: operators.Unsqueeze,

    // ------------- fused operations
    fused_Conv_Clip: fused_operators.Fused_Conv_Clip,
    fused_Conv_Relu: fused_operators.Fused_Conv_Relu,
    // fused_Conv_Sigmoid_Mul: fused_operators.Fused_Conv_Sigmoid_Mul, TODO
    fused_Dequant_Clip_Quant: fused_operators.Fused_Dequant_Clip_Quant,
    fused_Dequant_Pad_Quant_QLinConv: fused_operators.Fused_Dequant_Pad_Quant_QLinConv,
    fused_Dequant_Quant: fused_operators.Fused_Dequant_Quant,
    fused_Quant_Dequant: fused_operators.Fused_Quant_Dequant,
    fused_2Dequant_Add_Quant: fused_operators.Fused_2Dequant_Add_Quant,
    fused_Pad_Conv: fused_operators.Fused_Pad_Conv,

    // ------------- others
    useless: operators.Useless,

    pub fn init(nodeProto: *NodeProto) !Op_union {
        const op_type = nodeProto.op_type;
        return switch (op_type) {
            .ADD => Op_union{ .add = try operators.Add.init(nodeProto) },
            .AVERAGEPOOL => Op_union{ .averagePool = try operators.AveragePool.init(nodeProto) },
            .BATCHNORMALIZATION => Op_union{ .batchNormalization = try operators.BatchNormalization.init(nodeProto) },
            .CAST => Op_union{ .cast = try operators.Cast.init(nodeProto) },
            .CEIL => Op_union{ .ceil = try operators.Ceil.init(nodeProto) },
            .CLIP => Op_union{ .clip = try operators.Clip.init(nodeProto) },
            .CONCAT => Op_union{ .concat = try operators.Concat.init(nodeProto) },
            .CONSTANT => Op_union{ .constant = try operators.Constant.init(nodeProto) },
            .CONV => Op_union{ .conv = try operators.Conv.init(nodeProto) },
            .CONVINTEGER => Op_union{ .convInteger = try operators.ConvInteger.init(nodeProto) },
            .DEQUANTIZELINEAR => Op_union{ .dequantizeLinear = try operators.DequantizeLinear.init(nodeProto) },
            .DIV => Op_union{ .div = try operators.Div.init(nodeProto) },
            .DYNAMICQUANTIZELINEAR => Op_union{ .dynamicQuantizeLinear = try operators.DynamicQuantizeLinear.init(nodeProto) },
            .ELU => Op_union{ .elu = try operators.Elu.init(nodeProto) },
            .EXP => Op_union{ .exp = try operators.Exp.init(nodeProto) },
            .FLATTEN => Op_union{ .flatten = try operators.Flatten.init(nodeProto) },
            .FLOOR => Op_union{ .floor = try operators.Floor.init(nodeProto) },
            .GATHER => Op_union{ .gather = try operators.Gather.init(nodeProto) },
            .GATHERND => Op_union{ .gatherND = try operators.GatherND.init(nodeProto) },
            .GELU => Op_union{ .gelu = try operators.Gelu.init(nodeProto) },
            .GEMM => Op_union{ .gemm = try operators.Gemm.init(nodeProto) },
            .GLOBALAVERAGEPOOL => Op_union{ .globalAveragePool = try operators.GlobalAveragePool.init(nodeProto) },
            .IDENTITY => Op_union{ .identity = try operators.Identity.init(nodeProto) },
            .LEAKYRELU => Op_union{ .leakyRelu = try operators.LeakyRelu.init(nodeProto) },
            .MATMUL => Op_union{ .matMul = try operators.MatMul.init(nodeProto) },
            .MAXPOOL => Op_union{ .maxPool = try operators.MaxPool.init(nodeProto) },
            .MIN => Op_union{ .min = try operators.Min.init(nodeProto) },
            .MUL => Op_union{ .mul = try operators.Mul.init(nodeProto) },
            .NEG => Op_union{ .neg = try operators.Neg.init(nodeProto) },
            .NONMAXSUPPRESSION => Op_union{ .nonMaxSuppression = try operators.NonMaxSuppression.init(nodeProto) },
            .ONEHOT => Op_union{ .oneHot = try operators.OneHot.init(nodeProto) },
            .PAD => Op_union{ .pad = try operators.Pad.init(nodeProto) },
            .POW => Op_union{ .pow = try operators.Pow.init(nodeProto) },
            .QGEMM => Op_union{ .qgemm = try operators.QGemm.init(nodeProto) },
            .QLINEARADD => Op_union{ .qlinearadd = try operators.QLinearAdd.init(nodeProto) },
            .QLINEARAVERAGEPOOL => Op_union{ .qlinearaveragepool = try operators.QLinearAveragePool.init(nodeProto) },
            .QLINEARCONCAT => Op_union{ .qlinearconcat = try operators.QLinearConcat.init(nodeProto) },
            .QLINEARCONV => Op_union{ .qlinearconv = try operators.QLinearConv.init(nodeProto) },
            .QLINEARGLOBALAVERAGEPOOL => Op_union{ .qlinearglobalaveragepool = try operators.QLinearGlobalAveragePool.init(nodeProto) },
            .QLINEARMATMUL => Op_union{ .qlinearmatmul = try operators.QLinearMatMul.init(nodeProto) },
            .QLINEARMUL => Op_union{ .qlinearmul = try operators.QLinearMul.init(nodeProto) },
            .QLINEARSOFTMAX => Op_union{ .qlinearsoftmax = try operators.QLinearSoftmax.init(nodeProto) },
            .QUANTIZELINEAR => Op_union{ .quantizeLinear = try operators.QuantizeLinear.init(nodeProto) },
            .REDUCEMEAN => Op_union{ .reduceMean = try operators.ReduceMean.init(nodeProto) },
            .RELU => Op_union{ .relu = try operators.Relu.init(nodeProto) },
            .RESHAPE => Op_union{ .reshape = try operators.Reshape.init(nodeProto) },
            .RESIZE => Op_union{ .resize = try operators.Resize.init(nodeProto) },
            .SHAPE => Op_union{ .shape = try operators.Shape.init(nodeProto) },
            .SIGMOID => Op_union{ .sigmoid = try operators.Sigmoid.init(nodeProto) },
            .SLICE => Op_union{ .slice = try operators.Slice.init(nodeProto) },
            .SOFTMAX => Op_union{ .softmax = try operators.Softmax.init(nodeProto) },
            .SPLIT => Op_union{ .split = try operators.Split.init(nodeProto) },
            .SQUEEZE => Op_union{ .squeeze = try operators.Squeeze.init(nodeProto) },
            .SQRT => Op_union{ .sqrt = try operators.Sqrt.init(nodeProto) },
            .SUB => Op_union{ .sub = try operators.Sub.init(nodeProto) },
            .TOPK => Op_union{ .topK = try operators.TopK.init(nodeProto) },
            .TANH => Op_union{ .tanh = try operators.Tanh.init(nodeProto) },
            .TRANSPOSE => Op_union{ .transpose = try operators.Transpose.init(nodeProto) },
            .UNSQUEEZE => Op_union{ .unsqueeze = try operators.Unsqueeze.init(nodeProto) },
            else => {
                std.debug.print("\n\nERROR: init() is not available for {any} operator!! \n Pay attention! It may be a fused operation\n", .{op_type});
                return error.OpNotAvailable_for_init;
            },
        };
    }

    pub fn get_output_shape(self: Op_union) ![]usize {
        switch (self) {
            .pad => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qgemm => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearadd => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearaveragepool => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearconcat => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearconv => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearglobalaveragepool => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearmatmul => |ptr| return ptr.get_output_shape(),
            .qlinearmul => |ptr| return ptr.get_output_shape(),
            .qlinearsoftmax => |ptr| return ptr.compute_output_shape(),
            .quantizeLinear => |ptr| return ptr.get_output_shape(),
            inline else => |ptr, tag| ptr.get_output_shape() catch |e| {
                std.debug.print("\n\nERROR: get_output_shape() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        }
    }

    pub fn get_output_tensors(self: Op_union) ![]*TensorZant {
        return switch (self) {
            .useless => |ptr| try ptr.get_output_tensors(),
            inline else => |ptr, tag| ptr.get_output_tensors() catch |e| {
                std.debug.print("\n\nERROR: get_output_tensors() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        };
    }

    pub fn get_input_tensors(self: Op_union) ![]*TensorZant {
        return switch (self) {
            .useless => |ptr| try ptr.get_input_tensors(),
            inline else => |ptr, tag| ptr.get_input_tensors() catch |e| {
                std.debug.print("\n\nERROR: get_input_tensors() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        };
    }

    pub fn get_memory_footprint(self: Op_union) !usize {
        const input_tensors = try self.get_input_tensors();
        const output_tensors = try self.get_output_tensors();

        var node_mem: usize = 0;
        for (input_tensors) |t| {
            node_mem += t.getSize() * @sizeOf(t.ty);
        }

        for (output_tensors) |t| {
            node_mem += t.getSize() * @sizeOf(t.ty);
        }

        return node_mem;
    }

    pub fn write_op(self: Op_union, writer: std.fs.File.Writer) !void {
        switch (self) {
            .split => |ptr| try ptr.write_op(writer), //not working! error: .FAULT => unreachable,
            .useless => |ptr| try ptr.write_op(writer),
            inline else => |ptr, tag| ptr.write_op(writer) catch |e| {
                std.debug.print("\n\nERROR: write_op() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        }
    }

    pub fn print(self: Op_union) !void {
        switch (self) {
            .useless => |ptr| try ptr.print(),
            inline else => |ptr, tag| ptr.print() catch |e| {
                std.debug.print("\n\nERROR: print() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        }
    }

    pub fn sobstitute_tensors(self: *Op_union, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        switch (self.*) {
            .useless => |*ptr| try ptr.sobstitute_tensors(old_tensor, new_tensor),
            .fused_Quant_Dequant, .fused_Dequant_Quant => {
                //do nothing
            },
            inline else => |*ptr, tag| ptr.sobstitute_tensors(old_tensor, new_tensor) catch |e| {
                std.debug.print("\n\nERROR: sobstitute_tensors() is not available for {s}!! \n\n", .{@tagName(tag)});
                return e;
            },
        }
    }

    /// Render the lower-level math operation for the operator.
    /// DEPRECATED, TO BE REMOVED IN FUTURE VERSIONS
    pub fn render_lower_math_op(self: Op_union, builder: *UOpBuilder) !void {
        switch (self) {
            .add => |ptr| ptr.render_lower(builder),
            .ceil => |ptr| ptr.render_lower(builder),
            .conv => |ptr| ptr.render_lower(builder),
            .identity => |ptr| ptr.render_lower(builder),
            .div => |ptr| ptr.render_lower(builder),
            .identity => |ptr| ptr.render_lower(builder),
            .matMul => |ptr| ptr.render_lower(builder),
            .maxPool => |ptr| ptr.render_lower(builder),
            .mul => |ptr| ptr.render_lower(builder),
            .neg => |ptr| ptr.render_lower(builder),
            .relu => |ptr| ptr.render_lower(builder),
            .reshape => |ptr| ptr.render_lower(builder),
            .tanh => |ptr| ptr.render_lower(builder),
            else => {
                std.debug.print("\n\nERROR: render_lower() is not available!! \n\n", .{});
                return error.print_notAvailable;
            },
        }
    }
};
