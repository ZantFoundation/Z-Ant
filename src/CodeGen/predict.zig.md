# Code Documentation: ONNX Model Code Generation

## Table of Contents

1. [Introduction](#introduction)
2. [Module Overview](#module-overview)
3. [Function Details](#function-details)
    * [writePredict](#writepredict)
    * [write_graphSerialization](#write_graphserialization)
    * [write_outputsInitialization](#write_outputsinitialization)
    * [write_OutputShape](#write_outputshape)
    * [write_constantTensor](#write_constanttensor)
    * [write_OutputTensor](#write_outputtensor)
    * [write_outputsResetMethod](#write_outputsresetmethod)
    * [write_checks](#write_checks)
    * [write_predictInitialization](#write_predictinitialization)
    * [writeOperation](#writeoperation)
    * [writeReturn](#writereturn)

<a name="introduction"></a>
## 1. Introduction

This document details the code responsible for generating Zig code to perform inference on ONNX models.  The generated code efficiently executes the model's computation graph, handling various tensor operations and data types.

<a name="module-overview"></a>
## 2. Module Overview

The code utilizes several imported modules:

| Module          | Description                                         |
|-----------------|-----------------------------------------------------|
| `std`            | Zig's standard library                              |
| `zant`           | ONNX runtime library                               |
| `codegen`        | Custom code generation utilities                   |
| `codegen_options` | Configuration options for code generation           |

The code defines functions to generate the `predict` function which takes model inputs and produces outputs, along with helper functions to manage tensors, handle constant values, and perform various checks.

<a name="function-details"></a>
## 3. Function Details

<a name="writepredict"></a>
### `writePredict(writer: std.fs.File.Writer) !void`

Writes the `predict` function to the provided file writer. This function is the main entry point for inference.

1. **`write_outputsInitialization(writer)`:** Initializes output tensors.
2. **`write_outputsResetMethod(writer)`:** Generates a function to reset output tensors to zero.
3. **`predict` function generation:** Generates the main `predict` function structure.  It takes input data, input shape, shape length, and a pointer to the result as arguments.
4. **Conditional Logging:** Includes a conditional logging statement based on `codegen_options.log`.
5. **`resetOutputTensors()` call:** Calls the reset function.
6. **`write_checks(writer)`:** Writes input validation checks.
7. **`write_predictInitialization(writer)`:** Initializes input tensor for prediction.
8. **`write_graphSerialization(writer)`:** Serializes and writes the computation graph.
9. **`writeReturn(writer)`:** Writes the return statement.


<a name="write_graphserialization"></a>
### `write_graphSerialization(writer: std.fs.File.Writer) !void`

Processes and writes the computation graph to the file writer. This function iteratively processes computable nodes until no more nodes can be computed.

1. **Iterative Processing:** Uses a `while` loop to process computable nodes.  `utils.getComputableNodes` identifies nodes ready for computation.
2. **Node Processing:** Iterates through computable nodes, calling `writeOperation` for each.  `utils.setOutputsReady` marks outputs as ready for subsequent computation.
3. **Output Node Identification:** The last processed node is considered the network output, its name and shape are stored.
4. **Output Name Check:** Verifies consistency of output names; an error is returned if the detected output name differs from a previously determined name.


<a name="write_outputsinitialization"></a>
### `write_outputsInitialization(writer: std.fs.File.Writer) !void`

Initializes output tensors in the generated code. It iterates through each node in the graph and handles output tensor initialization differently for constant and non-constant nodes.

1. **Constant Node Handling:** For constant nodes (`op_type == "Constant"` and no inputs), it calls `write_constantTensor` to write the constant tensor's data.
2. **Non-Constant Node Handling:** For non-constant nodes, it calls `write_OutputShape` to get the size and then `write_OutputTensor` to declare and initialize the output tensor.


<a name="write_outputshape"></a>
### `write_OutputShape(writer: std.fs.File.Writer, output: *ReadyTensor) !i64`

Generates code to define the shape of an output tensor. It calculates and returns the total size of the tensor.


<a name="write_constanttensor"></a>
### `write_constantTensor(writer: std.fs.File.Writer, readyNode: *const ReadyNode) !void`

Writes the definition and initialization of a constant tensor to the file writer.

1. **Attribute Extraction:** Extracts the "value" attribute from the node's `AttributeProto` to get the constant tensor data.
2. **Data Type Handling:** Handles different data types (`float_data`, `int64_data`, `raw_data`) within the constant tensor.
3. **Data Writing:** Writes the shape and data values of the tensor to the file.
4. **Tensor Initialization:** Generates code to initialize the tensor using `Tensor.fromConstBuffer`.


<a name="write_outputtensor"></a>
### `write_OutputTensor(writer: std.fs.File.Writer, name: []const u8, size: i64) !void`

Writes the declaration and initialization of a non-constant output tensor, initializing the tensor with zeros.


<a name="write_outputsresetmethod"></a>
### `write_outputsResetMethod(writer: std.fs.File.Writer) !void`

Generates a function that resets all output tensors to zero.  It includes conditional logging.


<a name="write_checks"></a>
### `write_checks(writer: std.fs.File.Writer) !void`

Generates code for input validation checks. It verifies that the input shape provided at runtime matches the expected shape.


<a name="write_predictinitialization"></a>
### `write_predictInitialization(writer: std.fs.File.Writer) !void`

Generates code to initialize the input tensor for prediction. This includes allocating memory, copying data from the input, and converting the shape to the correct type.


<a name="writeoperation"></a>
### `writeOperation(writer: std.fs.File.Writer, readyNode: *ReadyNode) !void`

Writes the code for a specific node's operation. It delegates the actual operation writing to the `mathGen` module.


<a name="writereturn"></a>
### `writeReturn(writer: std.fs.File.Writer) !void`

Writes the return statement of the `predict` function, returning the pointer to the output tensor. Includes conditional logging.

