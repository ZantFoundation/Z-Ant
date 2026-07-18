# HOW TO ADD MATHEMATICAL OPERATIONS 
In this document you will find the explanations on how to integrate a mathematical operation in Z-ant.
The process can be divided in two big parts. 

# First part

The first part is about creating both lean and non-lean versions of the operation, defining the output shape computation and updating ONNX generation for fuzzing.

## Step 1: Create the Mathematical Operation

### File Location
Create a new file in `src/core/Tensor/TensorMath`.

### Output shape computation
Implement the get_op_output_shape function:
- the only function in the whole codebase to compute the shape of the output tensor 
- Non-Lean Version calls this function to allocate the output tensor

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


## Step 2: Add Tests

Include unit tests for both the lean and non-lean versions of the operation, as well as for the output shape function.


## Step 3: Update ONNX Generation for Fuzzing

Modify `onnx_gen.py` to support generating ONNX files for the new operation:

- Ensure the operation name adheres to the ONNX specification or register it as a custom op.
- Update the fuzzing test generation logic.
- **Note:** Align your changes with how **Gemm**, **Div**, or **Conv** are handled.

# Second part
Now that we have correctly implemented the operation, we have to integrate it with code generation.  
Start by reading [op_template](op_template.md) guide.

### File location
Create a new file in `src/IR_zant/op_union/operators`

## Step 1: create the struct & initialize it
- Define a struct for the operation in the new file
- Include fields for inputs, outputs, and ONNX attributes.
- Implement an `init` function to initialize the struct from a `NodeProto`, parsing inputs, outputs, and attributes according to the ONNX specification.

## Step 2: utility functions
Implement utility functions in the struct such as:
- get_output_shape
- get_output_tensor

## Step 3: compute output shape
Implement a compute_output_shape function in the struct to compute the output shape based on input shapes and attributes.

- Reuse the shape computation logic from `src/core/Tensor/TensorMath`
- Update the output tensor’s shape

## step 4: write op 
Implement a write_op function in the struct to generate the operation’s implementation during code generation.
- Handle tensor naming and attribute conversion using utility functions
- Generate a call to the lean version of the operation in 
`src/core/Tensor/TensorMath`
- Follow the patterns used for the other operations
- Ensure that the generated code uses the updated error handling for the mathematical operation is failures, which should look like (following the `op_add` operator):
    ```zig
    _ = try writer.print(
            \\
            \\
            \\    tensMath.sum_tensors_lean({s}, {s}, {s}, {s}, &tensor_{s}) catch return {d};
        , .{
            self.input_A.ty.toString(),
            self.output_C.ty.toString(),
            tensor_A_string, // Input tensor A
            tensor_B_string, // Input tensor B
            try utils.getSanitizedName(self.output_C.name), // Output tensor C
            utils.getMathErrorReturn(), // Error code for math errors
        });
    ```
    where `utils.getMathErrorReturn()` is a function from [IR_zant/utils.zig](../../../../Z-Ant/src/IR_zant/utils.zig) that returns the appropriate error code for mathematical operation failures (-101 if the first operation failed, -102 if the second one failed, etc.).








