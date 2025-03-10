# Static Library Testing: Internal Code Documentation

[Linked Table of Contents](#linked-table-of-contents)

## Linked Table of Contents

* [Introduction](#introduction)
* [Data Structures](#data-structures)
* [Functions](#functions)
    * [`generate_random_input_data`](#generate_random_input_data)
    * [`create_log_function`](#create_log_function)
    * [`run_prediction`](#run_prediction)
* [Test Cases](#test-cases)
    * [`test "Static Library - Random data Prediction Test"`](#test-static-library---random-data-prediction-test)
    * [`test "Static Library - Wrong Input Shape"`](#test-static-library---wrong-input-shape)
    * [`test "Static Library - Empty Input"`](#test-static-library---empty-input)
    * [`test "Static Library - Wrong Number of Dimensions"`](#test-static-library---wrong-number-of-dimensions)
    * [`test "Static Library - User Prediction Test" (TODO)](#test-static-library---user-prediction-test-todo)


## Introduction

This document details the implementation of the static library tests.  The tests focus on validating the `predict` function within the `model.lib` from `model_options.zig`. The tests cover various scenarios, including random data prediction, incorrect input shapes, empty inputs, and an incorrect number of dimensions.  Future development will include a user prediction test.

## Data Structures

The code utilizes the following key data structures:

| Data Structure | Type          | Description                                                                |
|-----------------|-----------------|----------------------------------------------------------------------------|
| `input_shape`   | `[]u32`        | Array defining the dimensions of the input tensor.                         |
| `input_data`    | `std.ArrayList(f32)` | Dynamic array holding the input data for prediction.                     |
| `result_buffer` | `[10]f32`      | Fixed-size array to store the prediction results (currently assumes 10 classes). |
| `result`        | `[*]f32`       | Pointer to the `result_buffer`.                                          |


## Functions

### `generate_random_input_data`

This function, implicitly defined within the `"Static Library - Random data Prediction Test"`, generates random input data for the prediction.

1. **Calculate Input Size:** It first computes the total number of elements in the input tensor based on the `input_shape`.
2. **Initialize Array:** It creates a `std.ArrayList(f32)` to hold the input data, allocating memory dynamically.
3. **Generate Random Numbers:** It uses `std.rand.DefaultPrng` to generate random floating-point numbers between 0 and 100.0.  The seed for the random number generator is obtained using `std.posix.getrandom` for better randomness.
4. **Populate Array:** The generated random numbers are appended to the `input_data` array.

### `create_log_function`

This function, implicitly defined within the tests, creates a logging function compatible with the static library's `setLogFunction`. It uses a `struct` to wrap `std.debug.print` for the callconv(.C) requirement of the static library's logging function.

### `run_prediction`

This function, implicitly defined within each test, executes the prediction using `model.lib.predict`. It takes the following arguments:

| Argument          | Type              | Description                                           |
|----------------------|----------------------|-------------------------------------------------------|
| `@ptrCast(&input_data.items)` | `[*]const f32`     | Pointer to the input data array.                        |
| `@ptrCast(&input_shape)`     | `[*]const u32`     | Pointer to the input shape array.                        |
| `input_shape.len`        | `usize`            | Number of dimensions in the input tensor (4 in this case).  |
| `&result`             | `[*]f32`           | Pointer to the array where the prediction results will be stored. |

The function then calls `model.lib.predict` to perform the prediction.


## Test Cases

### `test "Static Library - Random data Prediction Test"`

This test generates random input data using `generate_random_input_data`, sets a logging function, and runs the prediction using `run_prediction`.  It then checks for errors (implicitly, by the lack of errors during execution). The output shape of the prediction is assumed to be 10.


### `test "Static Library - Wrong Input Shape"`

This test intentionally provides an incorrect input shape to the `model.lib.predict` function. It increments each dimension of the correct `model_input_shape` to create the incorrect shape. The test aims to check for error handling in the static library when presented with a mismatched input shape.  The test implicitly assesses the library's robustness against invalid input.


### `test "Static Library - Empty Input"`

This test uses empty input arrays (`input_data` and `input_shape`) to check for the static library's handling of empty inputs.


### `test "Static Library - Wrong Number of Dimensions"`

This test provides an input with an incorrect number of dimensions (1D instead of the expected 4D). It verifies the library's response to a dimensionality mismatch.

### `test "Static Library - User Prediction Test" (TODO)`

This test is marked as a TODO and is intended to allow users to provide custom input data and run predictions.  It is not yet implemented.
