# **CMSIS-NN Integration in op\_qlinearconv.zig**

The op\_qlinearconv.zig file implements the ONNX QLinearConv operation. The integration of CMSIS-NN is handled entirely within the qlinearconv\_lean function, which acts as a dispatcher to choose the optimal convolution implementation based on the build target and configuration flags.

## **1\. The Dispatch Function: qlinearconv\_lean**

The qlinearconv\_lean function contains the logic to conditionally call the CMSIS-NN optimized implementation.

pub fn qlinearconv\_lean(...) \!void {  
    const cmsis\_enabled \= comptime mod\_cmsis.cmsisUsed();  
    if (cmsis\_enabled) {  
        // ... call CMSIS-NN implementation ...  
    } else {  
        // ... call embedded Zig implementation ...  
    }  
}

### **Conditional Execution**

| Code Line | Mechanism | Purpose |
| :---- | :---- | :---- |
| const cmsis\_enabled \= comptime mod\_cmsis.cmsisUsed(); | Uses comptime (compile-time execution) to check the state of the cmsisUsed() function imported from mod\_cmsis.zig. | This decision is fixed at compilation time, eliminating runtime overhead. The result depends on both the build flag (-Denable\_CMSIS) and the target architecture (e.g., Cortex-M). |
| if (cmsis\_enabled) | If the CMSIS conditions are met, the code paths diverge. | **High-Performance Path:** The execution proceeds to use the C-based cmsis\_nn.qlinearconv wrapper. |
| else | If CMSIS is disabled (e.g., compiling for a host machine or without the flag). | **Fallback Path:** The execution proceeds to use the native Zig qlinearconv\_embedded\_lean implementation. |

## **2\. CMSIS-NN Function Call**

When cmsis\_enabled is true, the CMSIS-NN wrapper is imported and called:

const cmsis\_nn \= @import("../Cmsis/wrappers/cmsis\_nn.zig");  
return cmsis\_nn.qlinearconv(  
    // ... all tensor and parameter arguments ...  
);

This ensures that the highly optimized C implementation from the CMSIS-NN library is used when cross-compiling for a supported ARM target with the appropriate build flags. The cmsis\_nn.zig wrapper handles the necessary data transformation and calling convention to interface the Zig tensors with the underlying CMSIS-NN C functions.