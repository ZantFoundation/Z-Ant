# Why Operations Were Consolidated in IR_zant

This document explains the reasons behind the decision to consolidate mathematical operations into single files within `src/codegen/IR_zant/op_union/operators` in Z-Ant.

## Background

In the initial version of Z-Ant, the implementation of mathematical operations was fragmented  across multiple files: operation logic in `src/core/Tensor/TensorMath`, shape computation in `shape_handler.zig`, code generation in `math_handler.zig` ans so on. This distributed approach led to complexity, maintenance challenges, and inefficiencies in the workflow.

To address these issues, the second version consolidates each operation into a single file in `src/codegen/IR_zant/op_union/operators`. Each file contains a struct with all necessary functionality: initialization (`init`), utility functions (`get_output_shape`, `get_output_tensor`, `print`), output shape computation (`compute_output_shape`), and code generation (`write_op`).

## Results
This representation is a one-to-one mapping with ONNX, with the distinction that ONNX is a string-based representation, whereas this is a data structure—specifically, a graph. Representing ONNX as a graph data structure enables several key capabilities in the subsequent steps of the Z-Ant workflow:

- **Graph Improvement and Linearization**: The graph structure allows for efficient traversal and manipulation of operations, facilitating the transformation of the ONNX model into an optimized internal representation. This graph can is then going to be linearized.

- **Fusion for Optimization**: The structured representation simplifies the fusion of operations, such as combining `Add` and `Gelu` into `Add_Gelu`. By having all operation metadata in a single struct, Z-Ant can analyze and merge operations easily, reducing the number of nodes in the graph and enhancing performance.

## Conclusion
By consolidating operations into a graph-based structure in `IR_zant`, Z-Ant not only mirrors the ONNX model but also enhances its ability to perform complex transformations, optimize performance, and generate executable code, paving the way for a more efficient and maintainable workflow.
