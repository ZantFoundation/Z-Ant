# Core

The `core` module contains the **Tensor** data structure, Z-Ant's fundamental building block. Every other subsystem (IR, codegen, operators) depends on it.

## Files

| File | Purpose |
|------|---------|
| `tensor.zig` | Minimal `Tensor(T)` struct and `AnyTensor` type-erased union. Construction, access, indexing. |
| `utils.zig` | Utilities that operate on tensors: debug printing, layout conversion, stride helpers, benchmarking. |

## Tensor vs AnyTensor

`Tensor(T)` is a generic, comptime-parameterised struct over a numeric type `T` (`f32`, `i8`, `u8`, ...). It holds:

- `data: []T` -- flat storage
- `shape: []usize` -- dimensions (e.g. `{1, 3, 224, 224}` for a batch-1 RGB image)
- `allocator` -- the allocator that owns `data` and `shape`

`AnyTensor` is a tagged union over all supported `Tensor(T)` pointer types. It is used when the element type is only known at runtime (e.g. when parsing ONNX models that declare tensor types as integers).

```zig
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const AnyTensor = zant.core.tensor.AnyTensor;
```

## tensor.zig -- what's inside

Only the essential API lives here:

- **Construction** -- `init`, `fromArray`, `fromShape`, `fromConstBuffer`, `copy`
- **Destruction** -- `deinit`
- **Element access** -- `get`, `set`, `get_at`, `set_at`
- **Indexing** -- `flatten_index` (multi-dim indices to flat offset, with fast paths for 1-4D)
- **Reset** -- `setToZero`
- **Re-exports** -- `math_standard` (operator math functions), `math_lean`, `utils`

## utils.zig -- what's inside

Everything that is useful but not part of the minimal Tensor definition:

- **Layout conversion** -- `from_NCHW_to_NHWC`, `from_NHWC_to_NCHW`
- **Array conversion** -- `toArray` (tensor to nested Zig slices)
- **Index / stride helpers** -- `get_flat_index`, `getStrides`, `ensure_4D_shape`, `copy_data_recursive`
- **Debug printing** -- `info`, `print`, `printMultidim`, `info_metal`
- **Benchmarking** -- `flatten_index_original`, `benchmark_flatten_index`
- **Internal helpers** -- `calculateProduct`, `flattenArray`

### Usage

Utils functions are free functions, not methods. They take `comptime T` and a tensor pointer:

```zig
const tensor_utils = zant.core.tensor.utils;

// Print tensor contents
tensor_utils.print(f32, &my_tensor);

// Get strides (caller owns returned slice)
const strides = try tensor_utils.getStrides(f32, &my_tensor);
defer my_tensor.allocator.free(strides);

// Convert layout
const nhwc = try tensor_utils.from_NCHW_to_NHWC(f32, allocator, &nchw_tensor);
```

## Access paths

From anywhere that has the `zant` module:

```zig
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const AnyTensor = zant.core.tensor.AnyTensor;
const TensMath = zant.core.tensor.math_standard;
const tensor_utils = zant.core.tensor.utils;
```
