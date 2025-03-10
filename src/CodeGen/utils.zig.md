# ONNX Code Generation - Internal Documentation

## Table of Contents

1. [Getters](#getters)
2. [Setters](#setters)
3. [Booleans](#booleans)
4. [Printers](#printers)
5. [Data Type Management](#data-type-management)


## <a name="getters"></a>Getters

| Function Name             | Description                                                                                                | Return Type                      | Algorithm                                                                                                                                  | Error Handling                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `getType(data_type)`     | Returns the equivalent Zig type for a given ONNX data type.                                                 | `!type`                           | Simple switch statement mapping ONNX `DataType` enum values to their corresponding Zig types.                                                   | Returns `error.DataTypeNotAvailable` for unknown types. |
| `getTypeString(data_type)` | Returns the equivalent string representation of a Zig type for a given ONNX data type.                       | `![]const u8`                     | Similar to `getType`, but returns a string instead of a type.                                                                               | Returns `error.DataTypeNotAvailable` for unknown types. |
| `getSanitizedName(name)`  | Returns a sanitized version of a tensor name, removing all non-alphanumeric characters and converting to lowercase. | `![]const u8`                     | Iterates through the input string, replacing non-alphanumeric characters with underscores and converting alphanumeric characters to lowercase. | Returns an error if allocation fails.             |
| `getComputableNodes`     | Returns a list of nodes that are ready for computation (all inputs are ready).                               | `!std.ArrayList(*ReadyNode)`     | Iterates through the ready graph, checking if all inputs for each node are ready.  Adds ready nodes to a new ArrayList.                       | Returns `error.OutputReadyTooEarly` if an output is already ready. |
| `getConstantTensorDims` | Returns the dimensions of a constant tensor from a NodeProto.                                            | `![]const i64`                    | Checks if the node is a constant, then extracts the dimensions from the `attribute` field of the `NodeProto`.                             | Returns `error.NodeNotConstant` if the node isn't a constant; `error.ConstantTensorAttributeNotAvailable` if dimensions are unavailable. |


## <a name="setters"></a>Setters

| Function Name        | Description                                                                        | Return Type | Algorithm                                                                                                                            | Error Handling                     |
|-----------------------|------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `setOutputsReady`     | Marks output tensors of a completed node as ready in the tensor hash map.             | `!void`     | Sets the `ready` flag of the completed node and all its output tensors in the hash map.                                                  | Returns an error if a key is not available in the hash map. |


## <a name="booleans"></a>Booleans

| Function Name        | Description                                                                  | Return Type | Algorithm                                                                                                |
|-----------------------|------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------|
| `areAllInputsReady`   | Returns `true` if all inputs for a given node are ready; otherwise, returns `false`. | `bool`      | Iterates through the node's inputs and checks the `ready` flag for each.                               |
| `isComputed`         | Returns `true` if all inputs and outputs of a node are ready; otherwise, returns `false`. | `!bool`     | Iterates through the node's inputs and outputs and checks the `ready` flag for each.                               |
| `isInitializer`      | Checks if a given name matches the name of any initializer tensor.             | `bool`      | Iterates through the list of initializer tensors and compares names.                                     |


## <a name="printers"></a>Printers

| Function Name          | Description                                                                                                                  | Algorithm                                                                                                                                                                         |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `printNodeList`         | Prints a list of all nodes in the graph, including their input and output tensors and readiness status.                          | Iterates through the nodes, printing each node's name, input tensors, and output tensors, indicating whether each tensor is ready.                                                |
| `printComputableNodes` | Prints the list of nodes ready for computation, including their operation type, inputs, and outputs and their readiness status. | Iterates through computable nodes, printing details as above. Includes error handling within the loop to ensure that ready statuses are properly reported.                      |
| `printOperations`      | Prints a list of unique ONNX operations found in the graph.                                                                       | Uses a hash map to store unique operations encountered while iterating through graph nodes. Prints the keys of the hashmap which contain the unique operation types. |
| `printTensorHashMap`   | Prints all entries in the given `tensorHashMap`.                                                                            | Iterates through the hash map, printing the key (tensor name) and value (ReadyTensor struct) for each entry.                                                                       |


## <a name="data-type-management"></a>Data Type Management

| Function Name                     | Description                                                                                                                     | Return Type                      | Algorithm                                                                                                                                                                                | Error Handling                                                     |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `i64SliceToUsizeSlice(input)`     | Converts a slice of `i64` to a slice of `usize`, performing bounds checking.                                                    | `![]usize`                        | Iterates through the input slice, checking for negative values and values exceeding the maximum `usize` value. Converts valid values to `usize`.                                               | Returns `error.NegativeValue` or `error.ValueTooLarge` if invalid values are found. |
| `usizeSliceToI64Slice(input)`     | Converts a slice of `usize` to a slice of `i64`, performing bounds checking.                                                    | `![]const i64`                   | Iterates through the input slice, checking for values exceeding the maximum `i64` value. Converts valid values to `i64`.                                                               | Returns `error.ValueTooLarge` if invalid values are found.        |
| `toUsize(comptime T, value)`       | Converts an integer value of any type `T` to `usize`, performing bounds checking.                                                 | `!usize`                         | Checks if `T` is an integer type. If `T` is signed, checks for negative values. Checks if the value exceeds the maximum `usize` value. Converts the value to `usize`.                  | Returns `error.NegativeValue` or `error.ValueTooLarge` if invalid values are found. |
| `sliceToUsizeSlice(slice)`        | Converts a slice of integers or arrays of integers to a slice of `usize` with bounds checking and -1 handling.                  | `[]usize`                         | Handles various slice types (.int, .array, .pointer), checking bounds and converting to `usize`.  Treats -1 as `std.math.maxInt(usize)`.                                                  | Panics if memory allocation fails or invalid values other than -1 are found. |
| `u32ToUsize(input, size)`         | Converts a `u32` array to a `usize` array.                                                                                      | `![]usize`                        | Iterates through the input array, performing bounds checking and converting valid values to `usize`.                                                                                         | Returns `error.NegativeValue` or `error.ValueTooLarge` if invalid values are found. |
| `parseNumbers(input)`             | Parses a comma-separated string of numbers into a slice of `i64`.                                                             | `![]i64`                         | Splits the input string by commas, parses each substring as an `i64`, and appends it to an array.                                                                                  | Returns an error if parsing fails.                              |
| `i64SliceToUsizeArrayString(values)` | Converts a slice of `i64` to a string representation of a `usize` array suitable for Zig code.                        | `![]const u8`                     | Iterates through the input slice, converts each `i64` to a string, and concatenates them into a string representing a `usize` array literal.                                             | Returns an error if concatenation fails.                   |

