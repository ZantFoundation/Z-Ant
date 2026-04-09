# Adding a New Mathematical Operation to Z-Ant

This guide explains how to implement a new mathematical operation in the Z-Ant project. The process includes creating both lean and non-lean versions of the operation, defining the output shape computation, integrating with code generation, and updating ONNX generation for fuzzing. All mathematical behavior must strictly adhere to the official ONNX documentation.

## Step 1: Create the Mathematical Operation

### File Location
Create a new directory under `src/IR_zant/op_union/operators/op_<name>/` containing `op_<name>.zig`, `zant_<name>.zig`, and `utils_<name>.zig`.

### Lean Version
- Takes a tensor as input.
- Performs the operation without any checks (e.g., no bounds or type validation).
- Writes the result directly to a pre-allocated tensor (no return value).

### Non-Lean Version
- Takes a tensor as input.
- Includes validation checks (e.g., shape compatibility and data type).
- Returns a new tensor as output.

### Key Differences
- **Lean:** Optimized for performance in resource-constrained environments by omitting safety checks.
- **Non-Lean:** Provides safety checks and returns a newly allocated tensor, suitable for development and debugging.

### Mathematical Behavior
The operation must conform to the ONNX specification. **Note:** Reference existing functions such as **Gemm**, **Div**, or **Conv** for guidance on structure, error handling, and tensor manipulation.

## Step 2: Define the Output Shape Function

In `utils_<name>.zig`, implement a function to compute the output shape based on the input shapes:

- Follow the ONNX documentation for the correct shape computation (e.g., similar to MatMul or Conv).
- Handle edge cases, such as empty tensors, appropriately.
- **Note:** Use the output shape functions of **Gemm**, **Div**, or **Conv** as examples.

## Step 3: Add Tests

Include unit tests for both the lean and non-lean versions of the operation, as well as for the output shape function.

## Step 4: Integrate with Code Generation

### Compute Output Shape
Update the code generation logic in `src/codegen` to incorporate the new output shape computation.

### Write the Operation
Add a function in `src/codegen` to generate the operation’s implementation during code generation. Update the dispatch logic accordingly. Follow the patterns used for **Gemm**, **Div**, or **Conv**.

## Step 5: Update ONNX Generation for Fuzzing

Modify `onnx_gen.py` to support generating ONNX files for the new operation:

- Ensure the operation name adheres to the ONNX specification or register it as a custom op.
- Update the fuzzing test generation logic.
- **Note:** Align your changes with how **Gemm**, **Div**, or **Conv** are handled.

