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
| Add | [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | ✅ | [op_addition.zig](../src/codegen/IR_zant/op_union/operators/op_addition.zig) | ✅ | ✅ | ✅ | |
| AveragePool | [AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ✅ | [op_averagePool.zig](../src/codegen/IR_zant/op_union/operators/op_averagePool.zig) | ✅ | ✅ | ✅ | |
| BatchNormalization | [BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ✅ | [op_batchNormalization.zig](../src/codegen/IR_zant/op_union/operators/op_batchNormalization.zig) | 🔶 | ✅ | ✅ | |
| Cast | [Cast](https://onnx.ai/onnx/operators/onnx__Cast.html) | ✅ | [op_cast.zig](../src/codegen/IR_zant/op_union/operators/op_cast.zig) | 🔴 | ✅ | ✅ | |
| Ceil | [Ceil](https://onnx.ai/onnx/operators/onnx__Ceil.html) | ✅ | [op_ceil.zig](../src/codegen/IR_zant/op_union/operators/op_ceil.zig) | ✅ | ✅ | ✅ | |
| Clip | [Clip](https://onnx.ai/onnx/operators/onnx__Clip.html) | ✅ | [op_clip.zig](../src/codegen/IR_zant/op_union/operators/op_clip.zig) | ✅ | ✅ | ✅ | |
| Concat | [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html)| ✅ | [op_concatenate.zig](../src/codegen/IR_zant/op_union/operators/op_concatenate.zig) | ✅ | ✅ | ✅ | |
| Conv | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | ✅ | [op_convolution.zig](../src/codegen/IR_zant/op_union/operators/op_convolution.zig) | ✅ | ✅ | ✅ | |
| ConvInteger | [ConvInteger](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) | ✅ | 🔴 | 🔴 | ✅ | ✅ | IR exists, math missing |
| DequantizeLinear | [DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) | ✅ | [op_dequantizeLinear.zig](../src/codegen/IR_zant/op_union/operators/op_dequantizeLinear.zig) | ✅ | ✅ | ✅ | |
| Div | [Div](https://onnx.ai/onnx/operators/onnx__Div.html) | ✅ | [op_division.zig](../src/codegen/IR_zant/op_union/operators/op_division.zig) | ✅ | ✅ | ✅ | |
| DynamicQuantizeLinear | [DynamicQuantizeLinear](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) | ✅ | [op_DynamicQuantizeLinear.zig](../src/codegen/IR_zant/op_union/operators/op_DynamicQuantizeLinear.zig) | 🔴 | ✅ | ✅ | |
| Elu | [Elu](https://onnx.ai/onnx/operators/onnx__Elu.html) | ✅ | [op_elu.zig](../src/codegen/IR_zant/op_union/operators/op_elu.zig) | ✅ | ✅ | ✅ | |
| Exp | [Exp](https://onnx.ai/onnx/operators/onnx__Exp.html) | ✅ | [op_exp.zig](../src/codegen/IR_zant/op_union/operators/op_exp.zig) | 🔴 | ✅ | ✅ | |
| Flatten | [Flatten](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ✅ | [op_flatten.zig](../src/codegen/IR_zant/op_union/operators/op_flatten.zig) | ✅ | ✅ | ✅ | missing axis attribute |
| Floor | [Floor](https://onnx.ai/onnx/operators/onnx__Floor.html) | ✅ | [op_floor.zig](../src/codegen/IR_zant/op_union/operators/op_floor.zig) | ✅ | ✅ | ✅ | |
| Gather | [Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)  | ✅ | [op_gather.zig](../src/codegen/IR_zant/op_union/operators/op_gather.zig) | ✅ | ✅ | ✅ | |
| GatherND | [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html) | ✅ | [op_gathernd.zig](../src/codegen/IR_zant/op_union/operators/op_gathernd.zig) | 🔴 | ✅ | 🔴 | |
| Gelu | [Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html) | ✅ | [op_gelu.zig](../src/codegen/IR_zant/op_union/operators/op_gelu.zig) | ✅ | ✅ | ✅ | |
| Gemm | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ✅ | [op_gemm.zig](../src/codegen/IR_zant/op_union/operators/op_gemm.zig) | ✅ | ✅ | ✅ | |
| GlobalAveragePool | [GlobalAveragePool](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | ✅ | [op_globalAveragePool.zig](../src/codegen/IR_zant/op_union/operators/op_globalAveragePool.zig) | 🔴 | ✅ | ✅ | |
| Identity | [Identity](https://onnx.ai/onnx/operators/onnx__Identity.html) | ✅ | [op_identity.zig](../src/codegen/IR_zant/op_union/operators/op_identity.zig) | ✅ | ✅ | ✅ | |
| LeakyRelu | [LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html) | ✅ | [op_leaky_reLU.zig](../src/codegen/IR_zant/op_union/operators/op_leaky_reLU.zig) | ✅ | ✅ | ✅ | |
| MatMul | [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)  | ✅ | [op_mat_mul.zig](../src/codegen/IR_zant/op_union/operators/op_mat_mul.zig) | ✅ | ✅ | ✅ | |
| MaxPool | [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html)   | ✅ | [op_maxPool.zig](../src/codegen/IR_zant/op_union/operators/op_maxPool.zig) | ✅ | ✅ | ✅ | Missing ceil param |
| Mean | [Mean](https://onnx.ai/onnx/operators/onnx__Mean.html) | 🔴 | [op_mean.zig](../src/codegen/IR_zant/op_union/operators/op_mean.zig) | ✅ | 🔴 | ✅ | |
| Min | [Min](https://onnx.ai/onnx/operators/onnx__Min.html) | ✅ | [op_min.zig](../src/codegen/IR_zant/op_union/operators/op_min.zig) | 🔴 | ✅ | 🔴 | |
| Mul | [Mul](https://onnx.ai/onnx/operators/onnx__Mul.html) | ✅ | [op_multiplication.zig](../src/codegen/IR_zant/op_union/operators/op_multiplication.zig) | ✅ | ✅ | ✅ | |
| Neg | [Neg](https://onnx.ai/onnx/operators/onnx__Neg.html) | ✅ | [op_neg.zig](../src/codegen/IR_zant/op_union/operators/op_neg.zig) | ✅ | ✅ | ✅ | |
| NonMaxSuppression | [NonMaxSuppression](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) | ✅ | [op_nonmaxsuppression.zig](../src/codegen/IR_zant/op_union/operators/op_nonmaxsuppression.zig) | 🔴 | ✅ | 🔴 | |
| OneHot | [OneHot](https://onnx.ai/onnx/operators/onnx__OneHot.html) | ✅ | [op_oneHot.zig](../src/codegen/IR_zant/op_union/operators/op_oneHot.zig) | ✅ | ✅ | ✅ | Bug in available_ops.txt |
| Pad | [Pad](https://onnx.ai/onnx/operators/onnx__Pad.html) | ✅ | [op_pad.zig](../src/codegen/IR_zant/op_union/operators/op_pad.zig) | 🔴 | ✅ | ✅ | |
| QLinearAdd | [QLinearAdd](https://onnx.ai/onnx/operators/onnx__QLinearAdd.html) | ✅ | [op_qlinearadd.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearadd.zig) | 🔴 | ✅ | ✅ | |
| QLinearAveragePool | [QLinearAveragePool](https://onnx.ai/onnx/operators/onnx__QLinearAveragePool.html) | ✅ | [op_qlinearaveragepool.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearaveragepool.zig) | 🔴 | ✅ | ✅ | |
| QLinearConcat | [QLinearConcat](https://onnx.ai/onnx/operators/onnx__QLinearConcat.html) | ✅ | [op_qlinearconcat.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearconcat.zig) | 🔴 | ✅ | ✅ | |
| QLinearConv | [QLinearConv](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | ✅ | [op_qlinearconv.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearconv.zig) | 🔶 | ✅ | ✅ | Tested in `test_quant_op_convolution.zig` |
| QLinearGlobalAveragePool | [QLinearGlobalAveragePool](https://onnx.ai/onnx/operators/onnx__QLinearGlobalAveragePool.html) | ✅ | [op_qlinearglobalaveragepool.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearglobalaveragepool.zig) | 🔴 | ✅ | ✅ | |
| QLinearMatMul | [QLinearMatMul](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) | ✅ | [op_qlinearmatmul.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearmatmul.zig) | 🔶 | ✅ | ✅ | Tested in `test_quant_op_mat_mul.zig` |
| QLinearMul | [QLinearMul](https://onnx.ai/onnx/operators/onnx__QLinearMul.html) | ✅ | [op_qlinearmul.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearmul.zig) | 🔴 | ✅ | 🔴 | |
| QLinearSoftmax | [QLinearSoftmax](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSoftmax) | ✅ | [op_qlinearsoftmax.zig](../src/codegen/IR_zant/op_union/operators/op_qlinearsoftmax.zig) | 🔴 | ✅ | 🔴 | Contrib Op |
| QuantizeLinear | [QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) | ✅ | [op_quantizeLinear.zig](../src/codegen/IR_zant/op_union/operators/op_quantizeLinear.zig) | ✅ | ✅ | ✅ | |
| ReduceMean | [ReduceMean](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ✅ | [lib_reduction_math.zig](../src/codegen/IR_zant/op_union/operators/zant_math_standard.zig) | ✅ | ✅ | ✅ | |
| Relu | [Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) | ✅ | [op_reLU.zig](../src/codegen/IR_zant/op_union/operators/op_reLU.zig) | ✅ | ✅ | ✅ | |
| Reshape | [Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ✅ | [op_reshape.zig](../src/codegen/IR_zant/op_union/operators/op_reshape.zig)  | ✅ | ✅ | ✅ | |
| Resize | [Resize](https://onnx.ai/onnx/operators/onnx__Resize.html) | ✅ | [op_resize.zig](../src/codegen/IR_zant/op_union/operators/op_resize.zig) | ✅ | ✅ | ✅| |
| Shape | [Shape](https://onnx.ai/onnx/operators/onnx__Shape.html) | ✅ | [op_shape.zig](../src/codegen/IR_zant/op_union/operators/op_shape.zig)| ✅ | ✅ | ✅ | Tested but not supported by Onnx Python Generator |
| Sigmoid | [Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html) | ✅ | [op_sigmoid.zig](../src/codegen/IR_zant/op_union/operators/op_sigmoid.zig)  | ✅ | ✅ | ✅ | |
| Slice | [Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) | ✅ | [op_slice.zig](../src/codegen/IR_zant/op_union/operators/op_slice.zig) | ✅ | ✅ | ✅ | |
| Softmax | [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html) | ✅ | [op_softmax.zig](../src/codegen/IR_zant/op_union/operators/op_softmax.zig) | ✅ | ✅ | ✅ | |
| Split | [Split](https://onnx.ai/onnx/operators/onnx__Split.html) | ✅ | [op_split.zig](../src/codegen/IR_zant/op_union/operators/op_split.zig)| ✅ | ✅ | ✅ | |
| Sqrt | [Sqrt](https://onnx.ai/onnx/operators/onnx__Sqrt.html) | ✅ | [op_sqrt.zig](../src/codegen/IR_zant/op_union/operators/op_sqrt.zig) | ✅ | ✅ | ✅ | |
| Squeeze | [Squeeze](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | ✅ | [op_squeeze.zig](../src/codegen/IR_zant/op_union/operators/op_squeeze.zig) | ✅ | ✅ | ✅ | |
| Sub | [Sub](https://onnx.ai/onnx/operators/onnx__Sub.html) | ✅ | [op_subtraction.zig](../src/codegen/IR_zant/op_union/operators/op_subtraction.zig) | ✅ | ✅| ✅| |
| Tanh | [Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html) | ✅ | [op_tanh.zig](../src/codegen/IR_zant/op_union/operators/op_tanh.zig) | ✅ | ✅ | ✅ | |
| TopK | [TopK](https://onnx.ai/onnx/operators/onnx__TopK.html) | ✅ | [op_topk.zig](../src/codegen/IR_zant/op_union/operators/op_topk.zig) | 🔴 | ✅ | 🔴 | |
| Transpose | [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html)| ✅ | [op_transpose.zig](../src/codegen/IR_zant/op_union/operators/op_transpose.zig) | ✅ | ✅ | ✅ | |
| Unsqueeze | [Unsqueeze](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ✅ | [op_unsqueeze.zig](../src/codegen/IR_zant/op_union/operators/op_unsqueeze.zig)| ✅ | ✅ | ✅| |

### Quantized Math (Backend)
These operations are found in `src/IR_zant/op_union/operators/` and form the backbone of the QLinear operations.

| math op name | tensor math | tensor math tests |
| :------------: | :---------: | :-----------: |
| Quantize | [op_quantize.zig](../src/codegen/IR_zant/op_union/operators/op_quantize.zig) | ✅ |
| Dequantize | [op_dequantize.zig](../src/codegen/IR_zant/op_union/operators/op_dequantize.zig) | ✅ |
| Quantized Addition | [quant_op_addition.zig](../src/codegen/IR_zant/op_union/operators/quant_op_addition.zig) | 🔴 |
| Quantized Convolution | [quant_op_convolution.zig](../src/codegen/IR_zant/op_union/operators/quant_op_convolution.zig) | ✅ |
| Quantized GEMM | [quant_op_gemm.zig](../src/codegen/IR_zant/op_union/operators/quant_op_gemm.zig) | 🔶 |
| Quantized MatMul | [quant_op_mat_mul.zig](../src/codegen/IR_zant/op_union/operators/quant_op_mat_mul.zig) | ✅ |
| Quantized Pooling | [quant_op_pooling.zig](../src/codegen/IR_zant/op_union/operators/quant_op_pooling.zig) | ✅ |