# **CMSIS-NN Wrapper for QLinearConv (cmsis\_nn.zig)**

The cmsis\_nn.zig file provides a high-level wrapper around the C functions in the CMSIS-NN library, specifically implementing the qlinearconv operation. This wrapper is responsible for adapting Zant's tensor data (which uses NCHW format) into the format expected by CMSIS-NN (NHWC format, s8 data type).

## **1\. Data Preparation and Conversion**

CMSIS-NN utilizes optimized C functions that require specific data layouts and types, which often differ from the standard ONNX/Zant format.

### **Layout and Type Transformation**

| Operation | Requirement (Zant) | Requirement (CMSIS-NN) | Implementation in Zig |
| :---- | :---- | :---- | :---- |
| **Input Data** | NCHW (Batch, Channel, Height, Width) | NHWC (Batch, Height, Width, Channel) | The input tensor x is converted and reordered from NCHW to NHWC layout and cast from the original InputType (often u8) to i8. |
| **Data Range** | Quantized u8 (0 to 255\) | Quantized s8 (-128 to 127\) | An adjustment is made by subtracting 128 from u8 values during the conversion process. Zero points are similarly adjusted. |
| **Output Data** | NCHW | NHWC (i8) | The output buffer is allocated as a temporary NHWC i8 buffer. After the CMSIS call, the data is reordered back to NCHW and restored to the original InputType (u8) by adding 128\. |
| **Weights** | NCHW (Out, In, KH, KW) | Packed OHWI (Out, H, W, In) | Weights are repacked into a contiguous i8 buffer (w\_packed) in the OHWI (Output Channel, Height, Width, Input Channel) order, subtracting the channel-specific zero point. |

### **Quantization Parameter Calculation**

The wrapper calculates the per-channel re-quantization parameters needed by CMSIS-NN's fixed-point arithmetic:

$$\\text{Scale}\_{\\text{ratio}} \= \\frac{\\text{Scale}\_{\\text{input}} \\times \\text{Scale}\_{\\text{weight}}}{\\text{Scale}\_{\\text{output}}}$$  
This ratio is then converted into a 32-bit multiplier and a fixed-point shift using qlinearconvUtils.quantizeMultiplier, and stored in multipliers\_buf and shifts\_buf.

## **2\. CMSIS-NN Parameter Structure Setup**

The wrapper initializes several C structures (cmsis\_nn\_dims, cmsis\_nn\_conv\_params, cmsis\_nn\_per\_channel\_quant\_params) needed for the underlying C call:

* **input\_dims, output\_dims**: Define the dimensions of the tensors in the NHWC format.  
* **conv\_params**: Contains convolution parameters like stride, padding, dilation, and crucially, the **input/output offsets** (cmsis\_input\_offset, cmsis\_output\_offset) calculated from the adjusted zero points.  
* **quant\_params**: Points to the calculated multiplier and shift buffers.

## **3\. Handling Bias and Buffer Allocation**

1. **Bias Conversion:** The optional bias tensor is converted to a required i32 format buffer (bias\_converted) as expected by the CMSIS functions.  
2. **Work Buffer:** The function determines the required working buffer size (buffer\_size) using arm\_convolve\_wrapper\_s8\_get\_buffer\_size or arm\_depthwise\_conv\_wrapper\_s8\_get\_buffer\_size, allocates a dynamic buffer (dyn\_buffer), and sets up the cmsis\_nn\_context structure.

## **4\. Dispatching to the CMSIS-NN C Function**

The wrapper supports three different convolution types:

| Convolution Type | Condition | CMSIS-NN Function Called |
| :---- | :---- | :---- |
| **Depthwise** | group\_val \== in\_channels | arm\_depthwise\_conv\_wrapper\_s8 |
| **Regular** | group\_val \== 1 | arm\_convolve\_wrapper\_s8 |
| **Grouped** | group\_val \> 1 (and not depthwise) | arm\_convolve\_wrapper\_s8 (called iteratively per group) |

For standard grouped convolution, the input and output tensors are processed group-by-group. The relevant input channels for the current group are copied from the full NHWC buffer into a temporary grouped buffer, the CMSIS function is called, and the output is copied back into the temporary full output buffer.

# **Future CMSIS-NN Wrapper (\_qlinearconv)**

In the second part of the file cmsis_nn.zig the CMSIS-NN QLinearConv wrapper is designed for a future state where the Zant tensor library utilizes the **NHWC (Batch, Height, Width, Channel)** convention for input and output, and **OHWI (Output, Height, Width, Input)** for weights.

This alignment significantly simplifies the wrapper by eliminating the need for expensive NCHW to NHWC data reordering.

## **Key Simplifications (Compared to Current Wrapper)**

Because the input/output and weight tensors are assumed to be in the CMSIS-compatible memory layout, the following complex and time-consuming operations are no longer required:

| Feature | Current Wrapper (NCHW) | Future Wrapper (NHWC/OHWI) | Impact |
| :---- | :---- | :---- | :---- |
| **Input Reordering** | Explicit NCHW --> NHWC data transformation. | **Removed.** Input data is used directly in its NHWC order. | Performance gain from removing memory copy/reordering. |
| **Output Reordering** | Explicit NHWC --> NCHW data transformation. | **Removed.** Output data is written directly to the tensor in its NHWC order. | Simpler, faster finalization. |
| **Weight Repacking** | Complex reordering logic to pack from ONNX's C(out)/C(in)/H/W to OHWI. | **Simplified.** Weights are already in OHWI; | Cleaner weight preparation. |

## **Data Transformation Flow**

The primary role of this future wrapper is solely to handle the signed/unsigned data type conversion required by CMSIS:

1. **Input Data (NHWC):** The data is converted directly from InputType (e.g., u8) to i8 by subtracting the zero adjustment (-128 for u8 case) without changing the memory layout.  
2. **Weights (OHWI):** Weights are iterated in their existing OHWI order, adjusted by their channel-specific zero point, and clamped to i8.  
3. **CMSIS Execution:** The arm\_convolve\_wrapper\_s8 or arm\_depthwise\_conv\_wrapper\_s8 function is called using the provided pointers, which are already in the correct NHWC/OHWI layout.  
4. **Output Data (NHWC):** The resulting i8 output is converted back to the original InputType (e.g., u8) by adding the zero restore value (+128 for u8 case), and written directly back into the output tensor's buffer, maintaining the NHWC layout.

This approach provides the most direct and lowest-overhead path for utilizing the CMSIS-NN acceleration.