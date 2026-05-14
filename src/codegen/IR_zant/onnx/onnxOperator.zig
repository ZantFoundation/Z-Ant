const std = @import("std");

pub const OnnxOperator = enum {
    ABS,
    ADD,
    AVERAGEPOOL,
    BATCHNORMALIZATION,
    CAST,
    CEIL,
    CLIP,
    CONCAT,
    CONSTANT,
    CONV,
    CONVINTEGER,
    DEQUANTIZE,
    DEQUANTIZELINEAR,
    DIV,
    DYNAMICQUANTIZELINEAR,
    ELU,
    EXP,
    FLATTEN,
    FLOOR,
    GATHER,
    GATHERND,
    GELU,
    GEMM,
    GLOBALAVERAGEPOOL,
    IDENTITY,
    LEAKYRELU,
    LOG,
    LOOP,
    MATMUL,
    MAXPOOL,
    MIN,
    MUL,
    NEG,
    NONMAXSUPPRESSION,
    ONEHOT,
    PAD,
    POW,
    QGEMM,
    QLINEARADD,
    QLINEARAVERAGEPOOL,
    QLINEARCONCAT,
    QLINEARCONV,
    QLINEARGLOBALAVERAGEPOOL,
    QLINEARMATMUL,
    QLINEARMUL,
    QLINEARSOFTMAX,
    QUANTIZE,
    QUANTIZELINEAR,
    REDUCEMEAN,
    RELU,
    RESHAPE,
    RESIZE,
    SHAPE,
    SIGMOID,
    SLICE,
    SOFTMAX,
    SPLIT,
    SQRT,
    SQUEEZE,
    SUB,
    TANH,
    TOPK,
    TRANSPOSE,
    UNSQUEEZE,
    USELESS,
};

pub fn isQlinear(op_type: OnnxOperator) bool {
    return op_type == .QLINEARADD or op_type == .QLINEARAVERAGEPOOL or op_type == .QLINEARCONCAT or op_type == .QLINEARCONV or op_type == .QLINEARGLOBALAVERAGEPOOL or op_type == .QLINEARMATMUL or op_type == .QLINEARMUL or op_type == .QLINEARSOFTMAX;
}

pub fn fromString(op_type: []const u8) !OnnxOperator {
    if (std.mem.eql(u8, op_type, "Abs")) return .ABS;
    if (std.mem.eql(u8, op_type, "Add")) return .ADD;
    if (std.mem.eql(u8, op_type, "AveragePool")) return .AVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "BatchNormalization")) return .BATCHNORMALIZATION;
    if (std.mem.eql(u8, op_type, "Cast")) return .CAST;
    if (std.mem.eql(u8, op_type, "Ceil")) return .CEIL;
    if (std.mem.eql(u8, op_type, "Clip")) return .CLIP;
    if (std.mem.eql(u8, op_type, "Concat")) return .CONCAT;
    if (std.mem.eql(u8, op_type, "Constant")) return .CONSTANT;
    if (std.mem.eql(u8, op_type, "Conv")) return .CONV;
    if (std.mem.eql(u8, op_type, "ConvInteger")) return .CONVINTEGER;
    if (std.mem.eql(u8, op_type, "Dequantize")) return .DEQUANTIZE;
    if (std.mem.eql(u8, op_type, "DequantizeLinear")) return .DEQUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "Div")) return .DIV;
    if (std.mem.eql(u8, op_type, "DynamicQuantizeLinear")) return .DYNAMICQUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "Elu")) return .ELU;
    if (std.mem.eql(u8, op_type, "Exp")) return .EXP;
    if (std.mem.eql(u8, op_type, "Flatten")) return .FLATTEN;
    if (std.mem.eql(u8, op_type, "Floor")) return .FLOOR;
    if (std.mem.eql(u8, op_type, "Gather")) return .GATHER;
    if (std.mem.eql(u8, op_type, "GatherND")) return .GATHERND;
    if (std.mem.eql(u8, op_type, "Gelu")) return .GELU;
    if (std.mem.eql(u8, op_type, "Gemm")) return .GEMM;
    if (std.mem.eql(u8, op_type, "GlobalAveragePool")) return .GLOBALAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "Identity")) return .IDENTITY;
    if (std.mem.eql(u8, op_type, "LeakyRelu")) return .LEAKYRELU;
    if (std.mem.eql(u8, op_type, "Log")) return .LOG;
    if (std.mem.eql(u8, op_type, "MatMul")) return .MATMUL;
    if (std.mem.eql(u8, op_type, "MaxPool")) return .MAXPOOL;
    if (std.mem.eql(u8, op_type, "Min")) return .MIN;
    if (std.mem.eql(u8, op_type, "Mul")) return .MUL;
    if (std.mem.eql(u8, op_type, "Neg")) return .NEG;
    if (std.mem.eql(u8, op_type, "NonMaxSuppression")) return .NONMAXSUPPRESSION;
    if (std.mem.eql(u8, op_type, "OneHot")) return .ONEHOT;
    if (std.mem.eql(u8, op_type, "Pad")) return .PAD;
    if (std.mem.eql(u8, op_type, "Pow")) return .POW;
    if (std.mem.eql(u8, op_type, "QGemm")) return .QGEMM;
    if (std.mem.eql(u8, op_type, "QLinearAdd")) return .QLINEARADD;
    if (std.mem.eql(u8, op_type, "QLinearAveragePool")) return .QLINEARAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "QLinearConcat")) return .QLINEARCONCAT;
    if (std.mem.eql(u8, op_type, "QLinearConv")) return .QLINEARCONV;
    if (std.mem.eql(u8, op_type, "QLinearGlobalAveragePool")) return .QLINEARGLOBALAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "QLinearMatMul")) return .QLINEARMATMUL;
    if (std.mem.eql(u8, op_type, "QLinearMul")) return .QLINEARMUL;
    if (std.mem.eql(u8, op_type, "QLinearSoftmax")) return .QLINEARSOFTMAX;
    if (std.mem.eql(u8, op_type, "Quantize")) return .QUANTIZE;
    if (std.mem.eql(u8, op_type, "QuantizeLinear")) return .QUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "ReduceMean")) return .REDUCEMEAN;
    if (std.mem.eql(u8, op_type, "Relu")) return .RELU;
    if (std.mem.eql(u8, op_type, "Reshape")) return .RESHAPE;
    if (std.mem.eql(u8, op_type, "Resize")) return .RESIZE;
    if (std.mem.eql(u8, op_type, "Shape")) return .SHAPE;
    if (std.mem.eql(u8, op_type, "Sigmoid")) return .SIGMOID;
    if (std.mem.eql(u8, op_type, "Slice")) return .SLICE;
    if (std.mem.eql(u8, op_type, "Softmax")) return .SOFTMAX;
    if (std.mem.eql(u8, op_type, "Split")) return .SPLIT;
    if (std.mem.eql(u8, op_type, "Sqrt")) return .SQRT;
    if (std.mem.eql(u8, op_type, "Squeeze")) return .SQUEEZE;
    if (std.mem.eql(u8, op_type, "Sub")) return .SUB;
    if (std.mem.eql(u8, op_type, "Tanh")) return .TANH;
    if (std.mem.eql(u8, op_type, "TopK")) return .TOPK;
    if (std.mem.eql(u8, op_type, "Transpose")) return .TRANSPOSE;
    if (std.mem.eql(u8, op_type, "Unsqueeze")) return .UNSQUEEZE;
    if (std.mem.eql(u8, op_type, "Useless")) return .USELESS;

    std.debug.print(" ERROR: operator {s} not supported ", .{op_type});

    return error.UnknownOperator;
}
