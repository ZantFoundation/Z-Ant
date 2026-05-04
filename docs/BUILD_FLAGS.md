# Z-Ant Build Configuration Flags

The Z-Ant build system is highly configurable. You can pass these flags to the `zig build` command using `-D[flag]=[value]`.

| Flag | Type | Default | Description | Relevant Steps |
|------|------|---------|-------------|----------------|
| **General Build Options** | | | | |
| `-Dtarget` | string | `"native"` | Target architecture (e.g., `thumb-freestanding`, `x86_64-linux`) | All |
| `-Dcpu` | string | `null` | CPU model (e.g., `cortex_m33`, `cortex_m7`) | All |
| `-Doptimize` | enum | `Debug` | Optimization level (`Debug`, `ReleaseSafe`, `ReleaseFast`, `ReleaseSmall`) | All |
| `-Dtrace_allocator` | bool | `true` | Use a tracing allocator for memory debugging | All |
| `-Dallocator` | string | `"raw_c_allocator"` | Underlying allocator to use | All |
| **Codegen & Model Options** | | | | |
| `-Dmodel` | string | `"mnist-8"` | Name of the model to process | `lib-gen`, `lib-exe`, `lib`, `lib-test` |
| `-Dmodel_path` | string | `datasets/...` | Path to the ONNX model file. Defaults to `datasets/models/{model}/{model}.onnx` | `lib-gen`, `lib-exe` |
| `-Dgenerated_path` | string | `generated/{model}/` | Output directory for generated code | `lib-gen`, `lib-exe`, `lib` |
| `-Doutput_path` | string | `""` | Custom output directory for the compiled static library | `lib` |
| `-Dshape` | string | `""` | Input tensor shape override (e.g., "1,3,224,224") | `lib-gen`, `lib-exe` |
| `-Dtype` | string | `"f32"` | Input tensor data type | `lib-gen`, `lib-exe` |
| `-Doutput_type` | string | `"f32"` | Output tensor data type | `lib-gen`, `lib-exe` |
| `-Dcomm` | bool | `false` | Generate code with comments included | `lib-gen`, `lib-exe` |
| `-Ddynamic` | bool | `true` | Enable dynamic memory allocation | `lib-gen`, `lib-exe` |
| `-Dstatic_planning` | string | `disabled` | Use with `-Ddynamic=false` to generate a compile-time memory plan. Simply use `enabled` to enable static planning. Other options are `pressure_then_size`, `pressure_then_liveness`, `liveness_first`, `size_first`, `first_step`, and their inverse variants (append `_inverse_first_step`; not valid with `enabled`). See `heuristics_for_static_planning.md` for more details. | `lib-gen`, `lib-exe` |
| `-Dfuse` | bool | `false` | Enable Kernel fusion optimization | `lib-gen`, `lib-exe` |
| `-Ddo_export` | bool | `false` | Generate exportable functions (for shared libs/FFI) | `lib-gen`, `lib-exe` |
| `-Dv` | string | `"v1"` | Codegen version to use | `lib-gen`, `lib-exe` |
| `-Dlog` | bool | `false` | Enable verbose logging during generation | `lib-gen`, `lib-exe` |
| `-Dxip` | bool | `false` | Enable XIP (Execute In Place) for neural network weights | `lib-gen`, `lib-exe` |
| `-Duse_tensor_pool` | bool | `false` | Allocate large tensor arrays to a specific `tensor_pool` section | Embedded targets |
| **Testing & Benchmarking** | | | | |
| `-Denable_user_tests` | bool | `false` | Generate user-defined test code | `lib-gen`, `lib-exe` |
| `-Dop` | string | `"all"` | Limit testing to a specific operator name | `test` |
| `-Dheavy` | bool | `false` | Run heavy/long-running tests | `test` |
| `-Dtest_name` | string | `""` | Specify a specific test case name to run | `test` |
| `-Dfull` | bool | `false` | Run the full benchmark suite | `benchmark` |
### Common Commands

* **Generate Library:** `zig build lib-gen -Dmodel=my_model`
* **Generate a static memory plan:** `zig build lib-gen -Dmodel=my_model -Ddynamic=false -Dstatic_planning=true`
* **Compile Static Lib:** `zig build lib -Dmodel=my_model -Dtarget=thumb-freestanding -Dcpu=cortex_m7`
* **Run Unit Tests:** `zig build test`
* **Run Benchmarks:** `zig build benchmark -Dfull=true`
