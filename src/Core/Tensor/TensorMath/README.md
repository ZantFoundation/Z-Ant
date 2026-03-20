# TensorMath Package

TensorMath includes all operators necessary to perform mathematical operations on tensors. When adding a new method, choose the proper `*_math.zig` file, or if you can't find a good one, create it yourself.

## op VS op_lean
The difference between these methods is that `op_lean` doesn't perform any checks and always returns void. The lean version is used only for MCU inference with ONNX format.
Each `*op_something.zig` implements both the standard version and the lean one.

**OSS**: lean_tensor_math is WIP

## op_something VS lib_something
`op_` contains only one method (operation) while `lib_` groups a cohort of methods. If you are implementing a complex functionality that doesn't match any `lib_`, create a new `op_yourMethod.zig`.

---

## Quantized Operators: `op_qlinear*.zig`

Files prefixed with `op_qlinear` implement the **ONNX `QLinear*` operator specification**.

### When to use this system
Use `op_qlinear*.zig` when:
- You are executing an **ONNX model** whose graph has explicit `QLinearAdd`, `QLinearMatMul`, `QLinearConv`, etc. nodes.
- Scale factors and zero points are **separate tensors** passed as inputs (matching the ONNX operator signature).
- You need **ONNX Runtime compatibility** — the float rescaling pipeline (`sum * a_scale * b_scale / y_scale`) is intentionally matched to ONNX Runtime's order of operations.

### Design contract
- Quantization parameters (`scale`, `zero_point`) are **NOT** embedded in the tensor. They are passed as separate `Tensor` arguments.
- Arithmetic is done in the **float domain**: inputs are dequantized to `f32`, added/multiplied, then requantized using the output scale/zero_point.
- All ops follow the standard 3-function pattern: `qlinearop()`, `lean_qlinearop()`, `get_qlinearop_output_shape()`.

### Available operators
| File | ONNX Op |
|---|---|
| `op_qlinearadd.zig` | QLinearAdd |
| `op_qlinearmatmul.zig` | QLinearMatMul |
| `op_qlinearconv.zig` | QLinearConv |
| `op_qlinearmul.zig` | QLinearMul |
| `op_qlinearconcat.zig` | QLinearConcat |
| `op_qlinearaveragepool.zig` | QLinearAveragePool |
| `op_qlinearglobalaveragepool.zig` | QLinearGlobalAveragePool |
| `op_qlinearsoftmax.zig` | QLinearSoftmax |
| `op_qlinearconv_simd.zig` | QLinearConv (SIMD-optimized variant) |
| `op_DynamicQuantizeLinear.zig` | DynamicQuantizeLinear |

> **Do NOT use these for new internal quantized kernels.** For performance-critical embedded inference kernels where quant params are embedded in the tensor, see [`../QuantTensorMath/README.md`](../QuantTensorMath/README.md).

---

# How to add a math operator
For each mathematical operation you want to implement you must write 3 pub methods:
- `operator()`
- `operator_lean()`
- `get_operator_output_shape()`

They must have the following structure:

Standard version of the function, used for unit tests, returns a Tensor.
```zig
pub fn operator(input: Tensor(T), attributes...) !Tensor(T) {

    // checks
    ...

    // compute output
    const output_shape = get_operator_output_shape(...);
    var output_tensor = try Tensor(T).fromShape(allocator, output_shape);

    try operator_lean(input, output, attributes...);

}
```
The lean version of the function, used during NN output prediction, returns void, no Tensor allocation inside the method.
```zig
pub fn operator_lean(input: Tensor(T), output: Tensor(T), attributes...) !void {
    // Actual computation of the output.
    // Dynamic allocation is forbidden here!
    // At this point you must assume that all the shapes are correct
    ...
}
```
A method to compute the shape of the output tensor. This must be the **ONLY** place in the code where the output shape is computed — avoid boilerplate code at all costs.
```zig
pub fn get_operator_output_shape(...) ![]usize {
    // Given the args computes the shape of the output
    ...
}
```