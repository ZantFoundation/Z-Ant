# **CMSIS-NN Wrapper for QLinearConv (cmsis_nn.zig)**

The `cmsis_nn.zig` file provides a high-level optimized wrapper around the C functions in the CMSIS-NN library. It specifically implements the `qlinearconv` operation, adapting Zant's tensor data to the requirements of the ARM CMSIS-NN kernels.

## **1. Data Preparation and Layouts**

CMSIS-NN utilizes optimized C functions that require specific data layouts and types. This wrapper bridges the gap between the Zant/ONNX definitions and CMSIS requirements.

### **Layout and Type Handling**

| Operation | Input/Output Requirement | Implementation Strategy |
| :--- | :--- | :--- |
| **Input Layout** | **NHWC** (Batch, Height, Width, Channel) | The wrapper assumes the input tensor `x` is already in **NHWC** format. No reordering is performed. |
| **Weight Layout** | **OHWI** (Out, Height, Width, In) | Weights are assumed to be in **OHWI** format. |
| **Data Types** | `s8` (Signed 8-bit) | CMSIS kernels operate on `s8`. <br>• If input is `i8`: **Zero-Copy** (passed directly).<br>• If input is `u8`: Converted to `s8` via SIMD subtraction ($x - 128$). |
| **Output Data** | `s8` (Signed 8-bit) | • If output tensor expects `i8`: Written directly (**Zero-Copy**).<br>• If output expects `u8`: Written to scratch buffer, then converted back via SIMD ($y + 128$). |

### **Quantization Parameter Calculation**

The wrapper calculates the per-channel re-quantization parameters needed by CMSIS-NN's fixed-point arithmetic:

$$\text{Scale}_{\text{ratio}} = \frac{\text{Scale}_{\text{input}} \times \text{Scale}_{\text{weight}}}{\text{Scale}_{\text{output}}}$$

This ratio is converted into a 32-bit integer multiplier and a bit-shift using `qlinearconvUtils.quantizeMultiplier`, stored in `multipliers_buf` and `shifts_buf`.

## **2. Performance Optimizations**

This implementation includes several architectural optimizations to minimize latency and memory overhead.

### **A. Memory Management (Arena)**
Instead of multiple individual allocations, a temporary `std.heap.ArenaAllocator` is initialized at the start of the function. All scratch buffers (bias conversions, packed weights, quantization arrays) are allocated from this arena, ensuring fast allocation and a single cleanup step at the end of execution.

### **B. Zero-Copy Paths**
The wrapper analyzes the input data types and quantization parameters to avoid memory copies whenever possible:
1.  **Bias:** If the bias tensor is already `i32`, the pointer is passed directly to CMSIS.
2.  **Weights:** If weights are `i8`, symmetric (zero-point is 0), and not required to be transposed for Depthwise ops, the underlying data pointer is used directly.
3.  **Input/Output:** If the tensor types match the CMSIS kernel requirements (`i8`), the wrapper writes directly to the tensor buffers.

### **C. SIMD Vectorization**
When Zero-Copy is not possible (e.g., converting `u8` inputs to `s8`), the wrapper uses Zig's `@Vector(16, ...)` primitives. This allows the CPU to process 16 bytes per instruction (128-bit SIMD), significantly speeding up the offset adjustment process:
* **Input:** `vector_u8 - 128` (wrapping subtraction)
* **Weights:** `clamp(vector_weight - zero_point)`
* **Output:** `vector_s8 + 128`

## **3. CMSIS-NN Parameter Setup**

The wrapper initializes the necessary C structures for the underlying kernel call:

* **`input_dims`, `output_dims`**: Defined directly from the tensor shapes (rank 4).
* **`conv_params`**: Contains stride, padding, dilation, and the crucial **offset adjustments** (`cmsis_input_offset`, `cmsis_output_offset`) derived from the quantization zero points.
* **`quant_params`**: Points to the per-channel multiplier and shift buffers.
* **Context Buffer**: The wrapper calls `arm_*_get_buffer_size` to calculate the required scratch memory for the kernel and allocates it from the Arena.

## **4. Dispatching Logic**

The wrapper dynamically selects the most efficient CMSIS-NN kernel based on the convolution attributes:

| Convolution Type | Condition | CMSIS-NN Function | Logic Details |
| :--- | :--- | :--- | :--- |
| **Depthwise** | `group == in_channels` | `arm_depthwise_conv_wrapper_s8` | Optimized path for depthwise separable convolutions. Weights may be reordered slightly if not symmetric to match CMSIS expectations. |
| **Standard** | `group == 1` | `arm_convolve_wrapper_s8` | Standard dense convolution. Uses SIMD for weight packing if zero-copy is not applicable. |
| **Grouped** | `group > 1` | `arm_convolve_wrapper_s8` | Iterative approach. The input is split into groups, processed sequentially using the standard convolution kernel, and results are interleaved into the output buffer. |

### **Output Handling**
After the CMSIS function returns `ARM_CMSIS_NN_SUCCESS`:
1.  If the output tensor was `i8` (Zero-Copy), the operation is complete.
2.  If the output tensor is `u8`, the data in the temporary `s8` buffer is converted back to `u8` (adding 128) using vectorized loops and copied to the final destination.