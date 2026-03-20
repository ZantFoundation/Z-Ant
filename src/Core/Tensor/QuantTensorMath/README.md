# QuantTensorMath Package

This package implements **performance-critical quantized tensor operations** for embedded and MCU targets. It is distinct from the ONNX-spec `QLinear*` operators in `../TensorMath/`.

---

## When to use this system

Use `QuantTensorMath` when:
- You are writing or optimizing **internal inference kernels** for embedded/MCU targets (e.g. STM32N6).
- Quantization parameters (`scale_factor`, `zero_point`) are **embedded directly inside the tensor** via `tensor.details.quant`.
- You need **integer arithmetic pipelines** — computation stays in `i8`/`i32` with fixed-point multiplier+shift rescaling to avoid any `f32` operations on constrained hardware.
- You need **SIMD vectorization** or **cache-blocked tiling** for throughput.

---

## How this system differs from `TensorMath/op_qlinear*.zig`

| Aspect | `QuantTensorMath/` (this folder) | `TensorMath/op_qlinear*.zig` |
|---|---|---|
| **Purpose** | Embedded/MCU inference kernels | ONNX model graph execution |
| **Quant params location** | Embedded in `tensor.details.quant` | Separate `Tensor` arguments (ONNX spec) |
| **Arithmetic pipeline** | Integer: accumulate in `i32`/`i64`, fixed-point rescale via multiplier+shift | Float: dequantize → `f32` add/mul → requantize |
| **Performance** | SIMD-vectorized, cache-blocked tiling | Simple loops, ONNX Runtime compatibility priority |
| **ONNX compatibility** | Not directly: uses internal quant metadata | Yes: matches ONNX Runtime's order of operations |

> **Rule of thumb**: If the tensors already carry their own `quant` details and you are targeting embedded performance, use this folder. If you are mapping an ONNX graph node with explicit scale/zero_point input tensors, use `../TensorMath/op_qlinear*.zig`.

---

## Available operators

| File | Operation |
|---|---|
| `quant_op_addition.zig` | Element-wise addition (integer fast path + float fallback for mismatched scales) |
| `quant_op_mat_mul.zig` | Matrix multiplication (SIMD + cache-blocked variants) |
| `quant_op_convolution.zig` | Convolution |
| `quant_op_gemm.zig` | General matrix multiply (Gemm) |
| `quant_op_pooling.zig` | Pooling operations |
| `op_quantize.zig` | Quantize a float tensor to integer (min-max or symmetric) |
| `op_dequantize.zig` | Dequantize an integer tensor back to float |
| `quant_tensor_math_standard.zig` | Shared quantization utilities (`lean_quantize_minmax`, `lean_dequantize`, etc.) |

---

## How to add a new operator

Follow the same 3-function pattern as `TensorMath`, with one extra requirement — always read and propagate `tensor.details.quant`:

```zig
/// Standard version: allocates output, returns Tensor.
pub fn quant_my_op(comptime T: anytype, input: *const Tensor(T), ...) !Tensor(T) {
    // Validate inputs
    ...

    // Allocate output and propagate quant details
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    output.details = input.details; // carry scale/zero_point forward

    try quant_lean_my_op(T, input, ..., &output);
    return output;
}

/// Lean version: no allocation, used in model inference hot path.
/// Dynamic allocation is FORBIDDEN here.
pub fn quant_lean_my_op(comptime T: anytype, input: *const Tensor(T), ..., output: *Tensor(T)) !void {
    switch (output.details) {
        .quant => |*qd| {
            // Integer arithmetic using qd.scale_factor, qd.zero_point
            // input.details.quant.scale_factor, input.details.quant.zero_point
            ...
        },
        else => return TensorError.NotQuantizedTensor,
    }
}

/// Shape helper: the ONLY place output shape is computed.
pub fn get_quant_my_op_output_shape(...) ![]usize {
    ...
}
```

### Key rules
1. **Always switch on `output.details`** — reject non-`.quant` tensors with `TensorError.NotQuantizedTensor`.
2. **Stay in integer domain** when inputs are `int` types. Convert to `f32` only when scale factors differ and a dequantize→requantize path is unavoidable (see `quant_op_addition.zig` for the pattern).
3. **Update `output.details.quant`** at the end of the lean function to reflect the correct output `scale_factor` and `zero_point`.
4. **No dynamic allocation in lean functions.** Use stack-allocated buffers for small ranks (see the 4D stack array pattern in `quant_op_addition.zig`).
