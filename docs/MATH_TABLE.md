# Supported Operations Status

✅ : complete
🔶 : WIP
🔴 : missing

onnx reference : \[onnx_name\]\(URL to [ONNX docs](https://onnx.ai/onnx/operators/index.html)\)  
tensor math : \[ fileName.zig \]\( path to the method \)  
tensor math tests: ✅, 🔶, 🔴  
oneOp model generator: ✅ if the oneOpModel is created, remember to add the onnx name inside [available_op](all_link_here)  

| math op name | onnx reference | IR_graph | tensor math | tensor math tests | codegen | oneOp model generator (.py) | notes |
| :------------: | :------------: | :---------: | :---------: | :-----------: | :-------: | :--------: | :------------- |
| Add | [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | ✅ | [op_addition.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_addition.zig) | ✅ | ✅ | ✅ | |
| AveragePool | [AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ✅ | [op_averagePool.zig](../src/Core/Tensor/TensorMath/op_averagePool.zig) | ✅ | ✅ | ✅ | |
| BatchNormalization | [BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ✅ | [op_batchNormalization.zig](../src/Core/Tensor/TensorMath/op_batchNormalization.zig) | 🔶 | ✅ | ✅ | |
| Cast | [Cast](https://onnx.ai/onnx/operators/onnx__Cast.html) | ✅ | [op_cast.zig](../src/Core/Tensor/TensorMath/op_cast.zig) | 🔴 | ✅ | ✅ | |
| Ceil | [Ceil](https://onnx.ai/onnx/operators/onnx__Ceil.html) | ✅ | [op_ceil.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_ceil.zig) | ✅ | ✅ | ✅ | |
| Clip | [Clip](https://onnx.ai/onnx/operators/onnx__Clip.html) | ✅ | [op_clip.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_clip.zig) | ✅ | ✅ | ✅ | |
| Concat | [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html)| ✅ | [op_concatenate.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_concatenate.zig) | ✅ | ✅ | ✅ | |
| Conv | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | ✅ | [op_convolution.zig](../src/Core/Tensor/TensorMath/op_convolution.zig) | ✅ | ✅ | ✅ | |
| ConvInteger | [ConvInteger](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) | ✅ | 🔴 | 🔴 | ✅ | ✅ | IR exists, math missing |
| DequantizeLinear | [DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) | ✅ | [op_dequantizeLinear.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_dequantizeLinear.zig) | ✅ | ✅ | ✅ | |
| Div | [Div](https://onnx.ai/onnx/operators/onnx__Div.html) | ✅ | [op_division.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_division.zig) | ✅ | ✅ | ✅ | |
| DynamicQuantizeLinear | [DynamicQuantizeLinear](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) | ✅ | [op_DynamicQuantizeLinear.zig](../src/Core/Tensor/TensorMath/op_DynamicQuantizeLinear.zig) | 🔴 | ✅ | ✅ | |
| Elu | [Elu](https://onnx.ai/onnx/operators/onnx__Elu.html) | ✅ | [op_elu.zig](../src/Core/Tensor/TensorMath/op_elu.zig) | ✅ | ✅ | ✅ | |
| Exp | [Exp](https://onnx.ai/onnx/operators/onnx__Exp.html) | ✅ | [op_exp.zig](../src/Core/Tensor/TensorMath/op_exp.zig) | 🔴 | ✅ | ✅ | |
| Flatten | [Flatten](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ✅ | [op_flatten.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_flatten.zig) | ✅ | ✅ | ✅ | missing axis attribute |
| Floor | [Floor](https://onnx.ai/onnx/operators/onnx__Floor.html) | ✅ | [op_floor.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_floor.zig) | ✅ | ✅ | ✅ | |
| Gather | [Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)  | ✅ | [op_gather.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_gather.zig) | ✅ | ✅ | ✅ | |
| GatherND | [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html) | ✅ | [op_gathernd.zig](../src/Core/Tensor/TensorMath/op_gathernd.zig) | 🔴 | ✅ | 🔴 | |
| Gelu | [Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html) | ✅ | [op_gelu.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_gelu.zig) | ✅ | ✅ | ✅ | |
| Gemm | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ✅ | [op_gemm.zig](../src/Core/Tensor/TensorMath/op_gemm.zig) | ✅ | ✅ | ✅ | |
| GlobalAveragePool | [GlobalAveragePool](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | ✅ | [op_globalAveragePool.zig](../src/Core/Tensor/TensorMath/op_globalAveragePool.zig) | 🔴 | ✅ | ✅ | |
| Identity | [Identity](https://onnx.ai/onnx/operators/onnx__Identity.html) | ✅ | [op_identity.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_identity.zig) | ✅ | ✅ | ✅ | |
| LeakyRelu | [LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html) | ✅ | [op_leaky_reLU.zig](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_leaky_reLU.zig) | ✅ | ✅ | ✅ | |
| MatMul | [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)  | ✅ | [op_mat_mul.zig](../src/Core/Tensor/TensorMath/op_mat_mul.zig) | ✅ | ✅ | ✅ | |
| MaxPool | [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html)   | ✅ | [op_maxPool.zig](../src/Core/Tensor/TensorMath/op_maxPool.zig) | ✅ | ✅ | ✅ | Missing ceil param |
| Mean | [Mean](https://onnx.ai/onnx/operators/onnx__Mean.html) | 🔴 | [op_mean.zig](../src/Core/Tensor/TensorMath/op_mean.zig) | ✅ | 🔴 | ✅ | |
| Min | [Min](https://onnx.ai/onnx/operators/onnx__Min.html) | ✅ | [op_min.zig](../src/Core/Tensor/TensorMath/op_min.zig) | 🔴 | ✅ | 🔴 | |
| Mul | [Mul](https://onnx.ai/onnx/operators/onnx__Mul.html) | ✅ | [op_multiplication.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_multiplication.zig) | ✅ | ✅ | ✅ | |
| Neg | [Neg](https://onnx.ai/onnx/operators/onnx__Neg.html) | ✅ | [op_neg.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_neg.zig) | ✅ | ✅ | ✅ | |
| NonMaxSuppression | [NonMaxSuppression](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) | ✅ | [op_nonmaxsuppression.zig](../src/Core/Tensor/TensorMath/op_nonmaxsuppression.zig) | 🔴 | ✅ | 🔴 | |
| OneHot | [OneHot](https://onnx.ai/onnx/operators/onnx__OneHot.html) | ✅ | [op_oneHot.zig](../src/Core/Tensor/TensorMath/op_oneHot.zig) | ✅ | ✅ | ✅ | Bug in available_ops.txt |
| Pad | [Pad](https://onnx.ai/onnx/operators/onnx__Pad.html) | ✅ | [op_pad.zig](../src/Core/Tensor/TensorMath/op_pad.zig) | 🔴 | ✅ | ✅ | |
| QLinearAdd | [QLinearAdd](https://onnx.ai/onnx/operators/onnx__QLinearAdd.html) | ✅ | [op_qlinearadd.zig](../src/Core/Tensor/TensorMath/op_qlinearadd.zig) | 🔴 | ✅ | ✅ | |
| QLinearAveragePool | [QLinearAveragePool](https://onnx.ai/onnx/operators/onnx__QLinearAveragePool.html) | ✅ | [op_qlinearaveragepool.zig](../src/Core/Tensor/TensorMath/op_qlinearaveragepool.zig) | 🔴 | ✅ | ✅ | |
| QLinearConcat | [QLinearConcat](https://onnx.ai/onnx/operators/onnx__QLinearConcat.html) | ✅ | [op_qlinearconcat.zig](../src/Core/Tensor/TensorMath/op_qlinearconcat.zig) | 🔴 | ✅ | ✅ | |
| QLinearConv | [QLinearConv](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | ✅ | [op_qlinearconv.zig](../src/Core/Tensor/TensorMath/op_qlinearconv.zig) | 🔶 | ✅ | ✅ | Tested in `test_quant_op_convolution.zig` |
| QLinearGlobalAveragePool | [QLinearGlobalAveragePool](https://onnx.ai/onnx/operators/onnx__QLinearGlobalAveragePool.html) | ✅ | [op_qlinearglobalaveragepool.zig](../src/Core/Tensor/TensorMath/op_qlinearglobalaveragepool.zig) | 🔴 | ✅ | ✅ | |
| QLinearMatMul | [QLinearMatMul](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) | ✅ | [op_qlinearmatmul.zig](../src/Core/Tensor/TensorMath/op_qlinearmatmul.zig) | 🔶 | ✅ | ✅ | Tested in `test_quant_op_mat_mul.zig` |
| QLinearMul | [QLinearMul](https://onnx.ai/onnx/operators/onnx__QLinearMul.html) | ✅ | [op_qlinearmul.zig](../src/Core/Tensor/TensorMath/op_qlinearmul.zig) | 🔴 | ✅ | 🔴 | |
| QLinearSoftmax | [QLinearSoftmax](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSoftmax) | ✅ | [op_qlinearsoftmax.zig](../src/Core/Tensor/TensorMath/op_qlinearsoftmax.zig) | 🔴 | ✅ | 🔴 | Contrib Op |
| QuantizeLinear | [QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) | ✅ | [op_quantizeLinear.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_quantizeLinear.zig) | ✅ | ✅ | ✅ | |
| ReduceMean | [ReduceMean](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ✅ | [lib_reduction_math.zig](../src/Core/Tensor/TensorMath/lib_reduction_math.zig) | ✅ | ✅ | ✅ | |
| Relu | [Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) | ✅ | [op_reLU.zig](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_reLU.zig) | ✅ | ✅ | ✅ | |
| Reshape | [Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ✅ | [op_reshape.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_reshape.zig)  | ✅ | ✅ | ✅ | |
| Resize | [Resize](https://onnx.ai/onnx/operators/onnx__Resize.html) | ✅ | [op_resize.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_resize.zig) | ✅ | ✅ | ✅| |
| Shape | [Shape](https://onnx.ai/onnx/operators/onnx__Shape.html) | ✅ | [op_shape.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_shape.zig)| ✅ | ✅ | ✅ | Tested but not supported by Onnx Python Generator |
| Sigmoid | [Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html) | ✅ | [op_sigmoid.zig](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_sigmoid.zig)  | ✅ | ✅ | ✅ | |
| Slice | [Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) | ✅ | [op_slice.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_slice.zig) | ✅ | ✅ | ✅ | |
| Softmax | [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html) | ✅ | [op_softmax.zig](../src/Core/Tensor/TensorMath/lib_activation_function_math/op_softmax.zig) | ✅ | ✅ | ✅ | |
| Split | [Split](https://onnx.ai/onnx/operators/onnx__Split.html) | ✅ | [op_split.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_split.zig)| ✅ | ✅ | ✅ | |
| Sqrt | [Sqrt](https://onnx.ai/onnx/operators/onnx__Sqrt.html) | ✅ | [op_sqrt.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_sqrt.zig) | ✅ | ✅ | ✅ | |
| Squeeze | [Squeeze](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | ✅ | [op_squeeze.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_squeeze.zig) | ✅ | ✅ | ✅ | |
| Sub | [Sub](https://onnx.ai/onnx/operators/onnx__Sub.html) | ✅ | [op_subtraction.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_subtraction.zig) | ✅ | ✅| ✅| |
| Tanh | [Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html) | ✅ | [op_tanh.zig](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_tanh.zig) | ✅ | ✅ | ✅ | |
| TopK | [TopK](https://onnx.ai/onnx/operators/onnx__TopK.html) | ✅ | [op_topk.zig](../src/Core/Tensor/TensorMath/op_topk.zig) | 🔴 | ✅ | 🔴 | |
| Transpose | [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html)| ✅ | [op_transpose.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_transpose.zig) | ✅ | ✅ | ✅ | |
| Unsqueeze | [Unsqueeze](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ✅ | [op_unsqueeze.zig](../src/Core/Tensor/TensorMath/lib_shape_math/op_unsqueeze.zig)| ✅ | ✅ | ✅| |

### Quantized Math (Backend)
These operations are found in `src/Core/Tensor/QuantTensorMath` and form the backbone of the QLinear operations.

| math op name | tensor math | tensor math tests |
| :------------: | :---------: | :-----------: |
| Quantize | [op_quantize.zig](../src/Core/Tensor/QuantTensorMath/op_quantize.zig) | ✅ |
| Dequantize | [op_dequantize.zig](../src/Core/Tensor/QuantTensorMath/op_dequantize.zig) | ✅ |
| Quantized Addition | [quant_op_addition.zig](../src/Core/Tensor/QuantTensorMath/quant_op_addition.zig) | 🔴 |
| Quantized Convolution | [quant_op_convolution.zig](../src/Core/Tensor/QuantTensorMath/quant_op_convolution.zig) | ✅ |
| Quantized GEMM | [quant_op_gemm.zig](../src/Core/Tensor/QuantTensorMath/quant_op_gemm.zig) | 🔶 |
| Quantized MatMul | [quant_op_mat_mul.zig](../src/Core/Tensor/QuantTensorMath/quant_op_mat_mul.zig) | ✅ |
| Quantized Pooling | [quant_op_pooling.zig](../src/Core/Tensor/QuantTensorMath/quant_op_pooling.zig) | ✅ |