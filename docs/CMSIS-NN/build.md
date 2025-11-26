# **CMSIS-NN Integration in build.zig**

The build.zig file ensures that CMSIS-NN support is conditionally applied to the necessary build artifacts (libraries, executables, and tests) when the \-Denable\_CMSIS=true flag is passed.

The core mechanism for this integration is the build\_utils.configureCmsisSupport() function, which links the necessary CMSIS-NN C source files and include directories to the target.

## **1\. Global Module Configuration**

At the start of the build function, CMSIS includes are configured for the main zant\_mod module itself. This allows Zant's internal modules (like cmsis_nn.zig and operation wrappers) to find the CMSIS header files.

| Code Section | Purpose |
| :---- | :---- |
| if (zantBuild.zantOptions.cmsis\_flags.enable\_cmsis) build\_utils.configureCmsisModuleIncludes(...) | Conditionally adds the CMSIS-NN header. This ensures that any Zig source file importing CMSIS can access the CMSIS definitions (e.g., when linking a C header). |

## **2\. Conditional CMSIS Support for Artifacts**

CMSIS support is conditionally applied to every build artifact that might utilize or test the generated neural network code.

The pattern is consistent across all relevant steps. This ensures that the CMSIS C sources are compiled and linked *only* when requested via the build flag *and* only for the specific target artifact.

### **Artifacts Using configureCmsisSupport**

| Function/Artifact | Type of Artifact | Role of CMSIS Integration |
| :---- | :---- | :---- |
| **unit\_test\_creation** | Root Unit Tests (e.g., test\_lib.zig) | Enables testing of base Zant utilities against an ARM target using CMSIS C libraries. |
| **lib\_codegen** | Code Generation Executable (codegen) | Links CMSIS support to the *generator* executable. While the code generator itself is typically run on the host, this ensures it has the context if needed and maintains consistent configuration across artifacts. |
| **lib\_exe** | Generated Model Executable (e.g., QLinearConv\_0\_exe) | Links the CMSIS C source files required for running the generated model on the host (for testing) or on the target device. |
| **lib\_test** | Generated Model Tests (e.g., test\_QLinearConv\_0.zig) | **Crucial for lib-test step.** Links the CMSIS library to the test runner, allowing the generated test file to execute CMSIS-NN functions and verify results against expected values. |
| **lib\_creation** | Generated Static Library (libzant.a) | Links the CMSIS C source files into the final static library artifact, making the optimized implementations available when the library is deployed to an embedded target. |
| **op\_codegen\_gen** | Single-Operator Code Generator | Links CMSIS support to the executable responsible for generating test code for single operations. |
| **op\_codegen\_test** | Single-Operator Unit Tests | Enables running tests for single-operator models using CMSIS-NN implementations. |
| **extractor\_gen / extractor\_test** | Extractor Tools and Tests | Enables CMSIS support for generating and testing extracted (fused) model nodes. |
| **benchmark\_create** | Benchmark Executable | Links CMSIS C source files to the benchmark tool, allowing performance profiling of the CMSIS-NN implementations. |
| **onnx\_parser** | ONNX Parser Tests | Links CMSIS support to the ONNX parser tests. |
