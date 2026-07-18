
# Style Guide for Zant

This document outlines the coding standards and best practices for contributing to Zant library written in Zig. Adhering to these guidelines will help maintain a cohesive and high-quality codebase.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Naming Conventions](#naming-conventions)
3. [Code Structure and Organization](#code-structure-and-organization)
4. [Formatting](#formatting)
5. [Comments and Documentation](#comments-and-documentation)
6. [Error Handling](#error-handling)
7. [Testing](#testing)
8. [Performance Considerations](#performance-considerations)
9. [Dependencies](#dependencies)


---

## 1. General Principles

- **Consistency**: Ensure uniformity across the codebase. When in doubt, refer to existing code patterns.

- **Clarity**: Write code that is easy to understand. Prioritize readability over cleverness.

- **Simplicity**: Avoid unnecessary complexity. Implement solutions in the simplest way possible.

- **Modularity**: Design components to be self-contained and reusable.

- **Documentation**: Keep code well-documented to facilitate understanding and collaboration.

## 2. Naming Conventions

Adhering to consistent naming conventions enhances code readability and maintainability.

### Types

- **Structs and Enums**: Use `TitleCase`.

  ```zig
  const NeuralNetwork = struct {
      // ...
  };

  const ActivationFunction = enum {
      ReLU,
      Sigmoid,
      Tanh,
  };
  ```

### Functions and Methods

- **Function Names**: Use `snake_case` as per Zig's conventions.

  ```zig
  fn train_model(model: *Model, data: DataSet) void {
      // ...
  }
  ```

### Variables and Constants

- **Variables**: Use `snake_case`.

  ```zig
  var learning_rate: f64 = 0.01;
  ```

- **Constants**: Use `SCREAMING_SNAKE_CASE`.

  ```zig
  const MAX_ITERATIONS: u32 = 1000;
  ```

### Generic Parameters

- **Generic Types**: Use `T`, `U`, `V`, etc., or descriptive names when clarity is needed.

  ```zig
  fn process_data(comptime T: type, data: []const T) void {
      // ...
  }
  ```

## 3. Code Structure and Organization

A well-organized codebase improves navigation and collaboration.

### Project Layout

- **Source Code**: Place in the `src/` directory.

- **Tests**: Place in the `src/tests/` directory

- **Documentation**: Place in the `docs/` directory.

## 4. Memory allocation policy
### Package allocator
A save memory usage ensures good performances. 
Always refer to the class `allocator.zig` to get the correct allocation. 
    
- Check that your class has the `YourClass_mod.addImport("pkgAllocator", allocator_mod);` inside build.zig.  
- Import the package allocator `const pkgAllocator = @import("pkgAllocator");`  
- when calling `try pkgAllocator.alloc(usize, something);` it automatically uses the proper allocator depending if we are running the main or the tests.

### Memory allocation and free policy
**" those who allocs, also frees "**  
So the class that manages the allocation/initialization of something, also manages the free/deinitialization.

## 5. Comments and Documentation

Proper commenting and documentation facilitate understanding and maintenance.

- **Function Documentation**: Use multiline comments to describe function purpose, parameters, and return values.

  ```zig
  /// Trains the model using the provided dataset.
  ///
  /// @param model The model to be trained.
  /// @param data The dataset for training.
  fn train_model(model: *Model, data: DataSet) void {
      // ...
  }
  ```

- **Inline Comments**: Use sparingly to explain complex logic.

  ```zig
  // Apply activation function
  output = activation(input);
  ```

- **TODO Comments**: Use `TODO:` to indicate areas for future improvement.

  ```zig
  // TODO: Optimize this loop for performance
  ```

## 6. Error Handling

Robust error handling ensures reliability.

- **Error Types**: Define specific error types for different failure modes.

  ```zig
  const Error = error{
      FileNotFound,
      InvalidData,
      // ...
  };
  ```

- **Error Propagation**: Use `try` to propagate errors.

  ```zig
  const file = try std.fs.openFile("data.csv", .{});
  ```

- **Error Handling**: Handle errors gracefully where appropriate.

  ```zig
  const result = process_data();
  switch (result) {
      Error.InvalidData => {
          // Handle invalid data error
      },
      else => |value| {
          // Process value
      }
  }
  ```

## 7. Testing

Comprehensive testing ensures code quality and correctness.

- **Test Functions**: Use `test "description"` syntax.

  ```zig
  test "< test-name >" {
    std.debug.print("\n     test: < test-name >", .{});
    // ...
  }
  ```
- **Test Coverage**: Aim for comprehensive coverage, including edge cases.

## 8. Performance Considerations

Optimizing performance is crucial for ML applications.

- **Memory Management**: Be mindful of allocations and deallocations. Use Zig's manual memory management features effectively.

- **Data Structures**: Choose appropriate data structures that offer optimal performance for the given task.

## 9. Dependencies

Manage dependencies carefully to ensure project stability.

- **Standard Library**: Prefer Zig's standard library for common functionalities.


By following this style guide, contributors can help maintain a consistent and high-quality codebase for the Zant ML library. Happy coding!
