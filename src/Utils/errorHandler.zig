const std = @import("std");

/// Los function errors
pub const LossError = error{
    SizeMismatch,
    ShapeMismatch,
    InvalidPrediction,
};

/// Layer errors
pub const LayerError = error{
    NullLayer,
    InvalidParameters,
    InvalidLayerType,
    Only2DSupported, //TODO: add description
    ZeroValueKernel, //TODO: add description
    ZeroValueStride, //TODO: add description
    FeatureNotSupported,
    TooLarge,
    LayerDimensionsInvalid,
    InputTensorWrongShape,
};

/// Type errors
pub const TypeError = error{
    UnsupportedType,
};

/// Architecture errors
pub const ArchitectureError = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};

/// Tensor Math errors
pub const TensorMathError = error{
    InvalidDimensions,
    MemError,
    InputTensorDifferentSize,
    InputTensorDifferentShape,
    InputTensorsWrongShape,
    OutputTensorDifferentSize,
    TooSmallOutputType,
    InputTensorDimensionMismatch,
    WrongStride,
    IncompatibleBroadcastShapes,
    EmptyTensorList,
    DivisionError,
    InvalidPadding,
    InvalidAxes,
    OutputTensorWrongShape,
    InvalidDataType,
    MismatchedDataTypes,
    Overflow,
    InvalidPaddingShape,
    InvalidInput,
    AxisOutOfRange,
    UnsupportedMode,
    InvalidPaddingSize,
    UnexpectedError,
    InputTensorNotScalar,
    InvalidGroupParameter,
    InvalidZeroPointShape,
    OutputShapeMismatch,
    ShapeMismatch,
    InvalidDepthShape,
    InvalidDepthValue,
    InvalidValuesShape,
    IndexOutOfBounds,
    InvalidShape,
    InvalidOutputLength,
    UnsupportedTensorDimensions,
    BroadcastingNotSupportedMixed,
};

/// Tensor errors
pub const TensorError = error{
    TensorNotInitialized,
    NotQuantizedTensor,
    InputArrayWrongType,
    InputArrayWrongSize,
    EmptyTensor,
    ZeroSizeTensor,
    NotOneHotEncoded,
    NanValue,
    NotFiniteValue,
    NegativeInfValue,
    PositiveInfValue,
    InvalidSliceIndices,
    InvalidSliceShape,
    SliceOutOfBounds,
    InvalidSliceStep,
    InvalidIndices,
    TooSmallToPadding,
    AxisOutOfBounds,
    MismatchedRank,
    MismatchedShape,
    IndexOutOfBounds,
    InvalidAxis,
    DuplicateAxis,
    InvalidInput,
    UnsupportedMode,
    UnsupportedDimension,
    InvalidSplitSize,
    InvalidRank,
    InvalidPermutation,
    ShapeMismatch,
    OutputTensorWrongShape,
};

/// A union type to represent any of the errors
pub const ErrorUnion = union(enum) {
    Loss: LossError,
    Layer: LayerError,
    Type: TypeError,
    Architecture: ArchitectureError,
    TensorMath: TensorMathError,
    Tensor: TensorError,
};

/// Function that returns the description of each error
/// #parameters:
///    myError: any error in this class, not ErrorUnion
/// #example of usage:
///    t1 and t2 Tensors,
///    _ = TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2) catch |err| {
///        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
///    };
pub fn errorDetails(myError: anyerror) []const u8 {
    return switch (myError) {
        //LOSSS
        LossError.SizeMismatch => "Loss: size mismatch between expected and actual",
        LossError.ShapeMismatch => "Loss: shape mismatch between tensors",
        LossError.InvalidPrediction => "Loss: invalid prediction value",

        //LAYER
        LayerError.NullLayer => "Layer: null layer encountered",
        LayerError.InvalidParameters => "Layer: invalid parameters specified",
        LayerError.TooLarge => "Layer: dimensions too large for memory allocation",
        LayerError.LayerDimensionsInvalid => "Layer: invalid dimensions for operation",
        LayerError.InputTensorWrongShape => "Layer: input tensor has wrong shape",

        //TYPE
        TypeError.UnsupportedType => "the Type you choose is not supported by this method/class",

        //ARCHITECTURE
        ArchitectureError.UnknownArchitecture => "Architecture: unknown architecture specified",
        ArchitectureError.UnderDevelopementArchitecture => "Architecture: architecture under development",

        //TENSORMATH
        TensorMathError.MemError => "TensorMath: memory error encountered",
        TensorMathError.InputTensorDifferentSize => "TensorMath: input tensor size mismatch",
        TensorMathError.InputTensorDifferentShape => "TensorMath: input tensor shape mismatch",
        TensorMathError.InputTensorsWrongShape => "TensorMath: input tensors have incompatible shapes",
        TensorMathError.OutputTensorDifferentSize => "TensorMath: output tensor size mismatch",
        TensorMathError.TooSmallOutputType => "TensorMath: output tensor type may lose information",
        TensorMathError.InputTensorDimensionMismatch => "TensorMath: input tensor dimension mismatch",
        TensorMathError.IncompatibleBroadcastShapes => "TensorMath: tensors have incompatible shapes for broadcasting",
        TensorMathError.InvalidDimensions => "TensorMath: invalid dimensions",
        TensorMathError.WrongStride => "TensorMath: wrong stride",
        TensorMathError.EmptyTensorList => "TensorMath: empty tensor list provided",
        TensorMathError.DivisionError => "TensorMath: division error encountered",
        TensorMathError.InvalidPadding => "TensorMath: invalid padding mode or values",
        TensorMathError.InvalidAxes => "TensorMath: invalid axes",
        TensorMathError.OutputTensorWrongShape => "TensorMath: output tensor has wrong shape",
        TensorMathError.InvalidDataType => "TensorMath: invalid data type",
        TensorMathError.MismatchedDataTypes => "TensorMath: mismatched data types between tensors",

        //TENSOR
        TensorError.TensorNotInitialized => "Tensor: tensor not initialized",
        TensorError.NotQuantizedTensor => "Tensor: tensor not quantized",
        TensorError.InputArrayWrongType => "Tensor: input array has wrong type",
        TensorError.InputArrayWrongSize => "Tensor: input array size mismatch",
        TensorError.EmptyTensor => "Tensor: empty tensor",
        TensorError.ZeroSizeTensor => "Tensor: tensor has zero size",
        TensorError.NotOneHotEncoded => "Tensor: tensor not one-hot encoded",
        TensorError.NanValue => "Tensor: NaN value in tensor",
        TensorError.NotFiniteValue => "Tensor: tensor has non-finite value",
        TensorError.NegativeInfValue => "Tensor: tensor has negative infinity value",
        TensorError.PositiveInfValue => "Tensor: tensor has positive infinity value",
        TensorError.InvalidSliceIndices => "Tensor: invalid slice indices",
        TensorError.InvalidSliceShape => "Tensor: invalid slice shape",
        TensorError.SliceOutOfBounds => "Tensor: slice out of bounds",
        TensorError.InvalidSliceStep => "Tensor: invalid slice step",
        TensorError.InvalidIndices => "Tensor: invalid indices",
        TensorError.TooSmallToPadding => "Tensor: too small to padding",
        TensorError.AxisOutOfBounds => "Tensor: axis out of bounds",
        TensorError.MismatchedRank => "Tensor: mismatched rank",
        TensorError.MismatchedShape => "Tensor: mismatched shape",
        TensorError.IndexOutOfBounds => "Tensor: index out of bounds",
        TensorError.InvalidAxis => "Tensor: invalid axis",
        TensorError.DuplicateAxis => "Tensor: contains duplicate axis",
        TensorError.InvalidInput => "Tensor: invalid input parameters for operation",
        TensorError.UnsupportedMode => "Tensor: unsupported interpolation mode",
        TensorError.UnsupportedDimension => "Tensor: operation not supported for this tensor dimension",
        TensorError.InvalidSplitSize => "Tensor: invalid split size for tensor dimension",
        TensorError.InvalidRank => "Tensor: the tensor rank is not suitable for operations",

        else => "Unknown error type",
    };
}
