# **CMSIS-NN Integration Utilities (utils.zig)**

The utils.zig file contains helper functions used by build.zig to configure the necessary C dependencies for the CMSIS-NN backend. It handles two primary tasks: finding include paths for Zig modules, and linking the required C source files to compilation steps.

## **1\. configureCmsisModuleIncludes**

This function is used to add CMSIS-NN, CMSIS-Core, and standard C library headers to a Zig module's list of import paths. This allows Zig code to correctly import and interface with C headers (e.g., in wrappers like cmsis\_nn.zig).

| Component | Path Configuration | Purpose |
| :---- | :---- | :---- |
| **CMSIS-NN** | Searches for third\_party/CMSIS-NN and its Include subdirectory. | Provides access to CMSIS-NN function prototypes. |
| **CMSIS-Core** | Searches for third\_party/CMSIS\_5/CMSIS/Core/Include. | Provides access to core CMSIS definitions, crucial for targeting Cortex-M microcontrollers. |
| **CMSIS-DSP** | Searches for third\_party/CMSIS-DSP/Include and PrivateInclude. | Provides access to underlying DSP routines often used by CMSIS-NN functions. |
| **ARM newlib** | Iterates through predefined arm\_none\_eabi\_paths. | Ensures standard C headers like \<string.h\> and \<math.h\> are found when cross-compiling for embedded ARM targets. |

## **2\. configureCmsisSupport**

This is the primary function for integrating CMSIS-NN into a specific build artifact (like a test executable or a static library). It performs two critical actions: setting include paths and linking the required C source files.

### **2.1 Include Path Configuration**

This step is similar to configureCmsisModuleIncludes, but it applies the include paths directly to the final compilation step (\*std.Build.Step.Compile). It relies on the assumption that the CMSIS libraries are located within the third\_party/ directory of the Zant repository.

### **2.2 Linking CMSIS-NN C Source Files**

To ensure that the generated Zig code, when running in CMSIS mode, can correctly call the optimized C functions, configureCmsisSupport explicitly links a set of required C source files.

**Convolution Functions (CMSIS-NN):**

A comprehensive list of convolution and depthwise convolution functions are linked, including essential files like:

* arm\_convolve\_s8.c and arm\_convolve\_get\_buffer\_sizes\_s8.c  
* Various wrappers (arm\_convolve\_wrapper\_s8.c) and specialized 1x1 convolutions.  
* Depthwise convolution implementations (arm\_depthwise\_conv\_wrapper\_s8.c, arm\_depthwise\_conv\_3x3\_s8, etc.).  
* Matrix multiplication kernels (arm\_nn\_mat\_mult\_kernel\_s8\_s16.c).

**Support Functions (CMSIS-NN):**

General utility functions used internally by the main operation kernels are also linked, such as:

* arm\_s8\_to\_s16\_unordered\_with\_offset.c  
* Various specialized matrix/vector multiplication routines.

**DSP Functions (CMSIS-DSP):**

Currently, this section explicitly links arm\_dot\_prod\_f32.c from the CMSIS-DSP library, ensuring that basic arithmetic operations required by some CMSIS-NN kernels are available.