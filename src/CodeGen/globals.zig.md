# ONNX Model Processing and Code Generation Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Data Structures](#2-data-structures)
    * [2.1 `io_struct`](#21-io_struct)
    * [2.2 `TensorTag`](#22-tensortagenum)
    * [2.3 `ReadyTensor`](#23-readytensor)
    * [2.4 `ReadyNode`](#24-readynode)
* [3. Global Variables](#3-global-variables)
* [4. Functions](#4-functions)
    * [4.1 `setGlobalAttributes`](#41-setglobalattributes)
    * [4.2 `populateReadyTensorHashMap`](#42-populatereadytensorhashmap)
    * [4.3 `addToTensorHashMap`](#43-addtotensorhashmap)
    * [4.4 `populateReadyGraph`](#44-populatereadygraph)


## 1. Introduction

This document details the internal workings of the Zig code responsible for processing ONNX models and generating code for computation.  The code utilizes the `zant` library for ONNX parsing and a custom `codegen` library for code generation (not shown here). The system focuses on preparing the ONNX graph for efficient computation by organizing tensors and nodes into readily accessible data structures.


## 2. Data Structures

### 2.1 `io_struct`

This struct defines the input/output structure for the network:

| Field      | Type        | Description                                      |
| ----------- |-------------|--------------------------------------------------|
| `name`     | `[]const u8` | Name of the input/output tensor.                 |
| `shape`    | `[]const i64`| Shape of the input/output tensor.                |


### 2.2 `TensorTag` enum

This enum categorizes tensors based on their role within the computational graph:

| Value       | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `INITIALIZER` | Tensor initialized as part of the model.                               |
| `CONSTANT`   | Constant tensor.                                                        |
| `INPUT`      | Input tensor to the network.                                             |
| `OUTPUT`     | Output tensor of the network.                                            |
| `LINK`       | Tensor acting as a connection between two nodes (output of one, input of another). |


### 2.3 `ReadyTensor`

This struct represents a tensor ready for computation:

| Field         | Type            | Description                                                                  |
|---------------|-----------------|------------------------------------------------------------------------------|
| `name`        | `[]const u8`    | Name of the tensor.                                                          |
| `ready`       | `bool`          | Indicates if the tensor is ready for computation (initially `true` for initializers and inputs). |
| `shape`       | `[]const i64`   | Shape of the tensor.                                                        |
| `tensorProto` | `?*TensorProto` | Pointer to the underlying `TensorProto` (might be `null` for inputs and links). |
| `tag`         | `TensorTag`     | Tag indicating the tensor's role (see `TensorTag` enum).                    |

The `createInitializer`, `createInput`, `createConstant`, and `createLink` functions are factory functions to create `ReadyTensor` instances with appropriate initialization.

### 2.4 `ReadyNode`

This struct represents a computational node, prepared for execution:

| Field        | Type                         | Description                                                                          |
|--------------|------------------------------|--------------------------------------------------------------------------------------|
| `nodeProto`  | `*NodeProto`                 | Pointer to the underlying `NodeProto`.                                                |
| `inputs`     | `std.ArrayList(*ReadyTensor)` | Array of input tensors for this node.                                                |
| `outputs`    | `std.ArrayList(*ReadyTensor)` | Array of output tensors for this node.                                               |
| `ready`      | `bool`                       | Indicates if the node is ready for computation (set to `false` initially).         |


The `create` function populates the `inputs` and `outputs` fields with references to corresponding `ReadyTensor` objects from `tensorHashMap`. It also calls `mathGen.compute_output_shape` to determine output tensor shapes. The `mathGen.compute_output_shape` function (not shown) is assumed to handle the logic for calculating output shapes based on node type and input shapes.


## 3. Global Variables

* `readyGraph`: An array of `ReadyNode` structs representing the prepared computational graph.
* `tensorHashMap`: A string hash map mapping tensor names (`TensorProto.name`) to `ReadyTensor` structs.
* `networkInput`:  Struct containing the name and shape of the network input.
* `networkOutput`: Struct containing the name and shape of the network output.
* `inputType`: Type of the input tensor (currently set to `f32`).


## 4. Functions

### 4.1 `setGlobalAttributes`

This function initializes global variables based on the provided ONNX model (`model`). It parses the input shape from command-line options (`codegen_options.shape`), sets `networkInput` and `networkOutput`, validates input shape consistency, and populates `tensorHashMap` and `readyGraph`.  Error handling is included for cases where input shape information is missing or inconsistent.  The function includes commented-out code that appears to be for more rigorous input shape validation but is currently disabled.

### 4.2 `populateReadyTensorHashMap`

This function populates the `tensorHashMap` with `ReadyTensor` instances for all tensors in the ONNX graph. It iterates through initializers, node inputs, and node outputs, creating and adding `ReadyTensor` objects based on their role (determined by `addToTensorHashMap`).

### 4.3 `addToTensorHashMap`

This function adds a `ReadyTensor` to the `tensorHashMap`.  It checks if the tensor already exists. If not, it determines the tensor's type (input, constant, or link) based on the name and creates the appropriate `ReadyTensor`.

### 4.4 `populateReadyGraph`

This function creates the `readyGraph` by iterating through the nodes of the ONNX graph and creating a `ReadyNode` for each, using the `ReadyNode.create` function.  This function leverages the already populated `tensorHashMap` to link nodes with their input and output tensors.
