# Z-Ant

<div align="left">
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zig-tests.yml/badge.svg" alt="Zig Tests" />
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zant-benchmarks.yml/badge.svg" alt="Zig Benchamrk Tests" />
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zig-codegen-tests.yml/badge.svg" alt="Zig Codegen Tests" />
</div>

![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)

## Project Overview

The deployment of deep neural networks on resource-constrained microcontrollers (MCUs) presents a fundamental engineering conflict. On one side, modern deep learning frameworks prioritize flexibility, utilizing dynamic memory allocation and runtime shape inference to support diverse model architectures. On the other side, MCU hardware (e.g., ARM Cortex-M, RISC-V) is characterized by hard constraints: typically less than 512\,KB of SRAM, limited Flash storage, and no Memory Management Unit (MMU).

When general-purpose tools are forced into this embedded environment, the mismatch results in bloated binaries, inefficient memory usage, and unpredictable execution jitter. Existing solutions often compromise on one of three pillars: performance, portability, or determinism.

**Zant** presents an open-source inference engine and compiler framework designed specifically to bridge this gap. Unlike interpreter-based systems, Zant resolves all graph scheduling, memory planning, and optimization decisions at compile time. It inputs standard ONNX models and outputs deterministic, static libraries written in Zig/C that are ready for integration into bare-metal firmware.

Read [ZANT_WORKFLOW](docs/ZANT_WORKFLOW.md), [ZANT CLI](docs/ZANT_CLI.md) and [BUILD_FLAGS](docs/BUILD_FLAGS.md) for a better understanding and more details!

## ✨ Why Z-Ant?

- **⚡ Microsecond** inference on ARM Cortex-M, RISC-V, x86
- **📦 Zero dependencies** - single static library deployment
- **🎯 ONNX native** - direct model deployment from ONNX
- **🔧 30+ operators** - comprehensive neural network support
- **📷 Built-in image processing** - JPEG decode + preprocessing
- **🧠 Smart optimization** - quantization, pruning, memory efficiency

---

## 🚀 Quick Start

Prerequisites

- [Zig 0.15.x](https://ziglang.org/learn/getting-started/) _(run `./scripts/install_zig.sh` to fetch a local copy; set `ZIG_DOWNLOAD_URL=file:///absolute/path/to/zig-linux-x86_64-0.15.2.tar.xz` when working from a pre-downloaded archive)_
- `qemu-system-arm` 7.2+ _(install via `./scripts/install_qemu.sh` or your platform package manager when running the STM32 N6 QEMU harness)_

### Get Started in 2 Minutes
```bash
# Clone and verify installation
git clone https://github.com/ZantFoundation/Z-Ant.git
cd Z-Ant

# Install Zig 0.15.2 locally (optional if you already have it)
./scripts/install_zig.sh
# (use ZIG_DOWNLOAD_URL or ZIG_DOWNLOAD_BASE to point at a mirror if needed)
export PATH="$(pwd)/.zig-toolchain/current:$PATH"

# - put your onnx model inside /datasets/models in a folder with the same of the model to to have: /datasets/models/my_model/my_model.onnx

# - simplify and prepare the model for zant inference engine
./zant input_setter --model my_model --shape your,model,sha,pe

# - Generate test data
./zant user_tests_gen --model my_model

# --- GENERATING THE Single Node lib and test it ---
#For a N nodes model it creates N onnx models, one for each node with respective tests.
./zant onnx_extract  --model my_model

#generate libs for extracted nodes
zig build extractor-gen -Dmodel="my_model"

#test extracted nodes
zig build extractor-test -Dmodel="my_model"

# --- GENERATING THE LIBRARY and TESTS ---
# Generate code for a specific model
zig build lib-gen -Dmodel="my_model" -Denable_user_tests [ -Ddo_export -Dlog -Dcomm ... ]

# Test the generated code
zig build lib-test -Dmodel="my_model" -Denable_user_tests [ -Ddo_export -Dlog -Dcomm ... ]

# Build the static library
zig build lib -Dmodel="my_model" [-Doptimize=Release? -Dtarget=... -Dcpu=...]

```

## Optional SDK downloads
```bash
# Fetch CMSIS-NN into third_party/CMSIS-NN
./scripts/fetch_cmsis_nn.sh
# (set CMSIS_NN_REPO or CMSIS_NN_REF to use a mirror/specific release)
# (set CMSIS_NN_ARCHIVE=/absolute/path/to/CMSIS-NN-main.zip to install from a local archive without network access)

# Fetch the Arm Ethos-U core driver
./scripts/fetch_ethos_u_driver.sh
# (set ETHOS_U_REPO / ETHOS_U_REF to use a fork or pinned revision)
# (set ETHOS_U_ARCHIVE=/absolute/path/to/ethos-u-driver.zip for offline installs)

# Install qemu-system-arm for the STM32 N6 QEMU regression harness
./scripts/install_qemu.sh
# (requires administrator privileges; set QEMU_SKIP_APT_UPDATE=1 to skip `apt-get update` on Debian/Ubuntu)
```

### Optional STM32 N6 accelerator testing
```bash
# Host smoke test for the C shim (builds reference/CMSIS/Ethos shared objects)
./scripts/test_stm32n6_conv.py

# Bare-metal regression harness executed inside QEMU
# Automatically discovers CMSIS-DSP / CMSIS-NN in third_party; pass --cmsis-include/--cmsis-nn-include to override.
./scripts/test_stm32n6_qemu.py --arm-prefix arm-none-eabi --repeat 3

# Sample output (PASS markers are emitted by the firmware, the harness exits immediately after the first PASS):
#   [run]   reference
#   stm32n6 reference PASS
#   ✅ reference completed in 40.57 ms
```

The script terminates QEMU as soon as the PASS banner appears, so the reported time reflects the actual firmware runtime instead of the former 3 s watchdog timeout. A non-zero exit status accompanied by `fatal: unexpected exception` indicates a crash inside the firmware before the PASS message is printed.

## 💼 Integration Examples

### CMake Integration

```cmake
target_link_libraries(your_project PUBLIC path/to/libzant.a)
```

### Arduino/Embedded C

```c
#include "lib_my_model.h"

// Optional: Set custom logging
extern void setLogFunction(void (*log_function)(uint8_t *string));

// Your inference code here
```

## 🎯 Real-World Examples

### Image Classification on Cortex-M33

```bash
# Generate optimized library for image classifier
zig build codegen -Dmodel=mobilenet_v2 -Dmodel_path=models/mobilenet.onnx
zig build lib -Dmodel=mobilenet_v2 -Dtarget=thumb-freestanding -Dcpu=cortex_m33 -Doutput_path=deployment/
```

### Multi-Platform Testing

```bash
# Test on different architectures
zig build test-generated-lib -Dmodel=my_model -Dtarget=native
zig build test-generated-lib -Dmodel=my_model -Dtarget=thumb-freestanding -Dcpu=cortex_m4
```

### Time-Series Classification via GAF/MTF Encoding

The `TensorToImage` module encodes a 1D time series into Gramian Angular Summation/Difference
Field and Markov Transition Field images, so a CNN model can be used for time-series
classification (e.g. ECG signals). See `examples/gaf-demo/README.md`, `src/TensorToImage/OVERVIEW.md`,
and `examples/Nicla-ecg/` for a full Arduino deployment example.

```bash
zig build gaf-demo -- examples/gaf-demo/sample.csv --colormap viridis
```

## 🛠️ Development

### For Contributors

```bash
# Run full test suite
zig build test --summary all

# Test heavy computational operations
zig build test -Dheavy=true

# Test specific operator implementations
zig build op-codegen-test -Dop=Conv

# Generate and test single operations
zig build op-codegen-gen -Dop=Add
```

### Project Structure

```
Z-Ant/
├── src/                   # Source code (3 Zig build modules)
│   ├── utils/             # Allocator facade (zant_utils module)
│   ├── codegen.zig        # Code generation entry point (codegen module)
│   ├── codegen/           # Code generation engine
│   │   ├── IR_zant.zig    # IR entry point (IR_zant module)
│   │   └── IR_zant/       # IR: graph, nodes, tensors, operators, fusion
│   │       ├── onnx.zig   # ONNX (no onnx module is present, it is imported via relative path inside IR_zant)
│   │       └── onnx/      # ONNX protobuf parser (custom parser, no external deps)
│   ├── TensorToImage/     # GASF/GADF/MTF time-series-to-image encoders (TensorToImage module)
│   └── main.zig           # CLI entry point for lib-gen
├── tests/                 # Comprehensive test suite
├── datasets/              # Sample models and test data
├── generated/             # Generated code output
├── zantBuild/             # Build system modules (options, flags, module wiring)
├── examples/              # Arduino and microcontroller examples
└── docs/                  # Documentation and guides
```

---

## 🛠️ CI/CD

- `zig-tests` – regression suite covering the core runtime.
- `zig-codegen-tests` – validates generated operators and glue code.
- `zant-benchmarks` – runs the Beer end-to-end benchmark and refreshes the metrics below.

<!-- BEER_TIMINGS_START -->
Beer model timing (QEMU, Cortex-M55):

- Reference: 859.70 ms
- CMSIS-NN: 855.59 ms
- Improvement: 4.11 ms (0.5%)
<!-- BEER_TIMINGS_END -->

## 🤝 Contributing

We welcome contributions from developers of all skill levels! Here's how to get involved:

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your work
4. **Make your changes** following our coding standards
5. **Run tests** to ensure everything works
6. **Submit a pull request** for review

### Ways to Contribute

- **🐛 Bug Reports**: Found an issue? Let us know!
- **✨ Feature Requests**: Have an idea? Share it with us!
- **💻 Code Contributions**: Improve the codebase or add new features
- **📚 Documentation**: Help make the project easier to understand
- **🧪 Testing**: Write tests or improve test coverage

### Community Guidelines

- Follow our [Code of Conduct](.github/CODE_OF_CONDUCT.md)
- Check out the [Contributing Guide](docs/CONTRIBUTING.md) for detailed guidelines
- Join discussions on GitHub Issues and Discussions

### Recognition

All contributors are recognized in our [Contributors list](https://github.com/ZIGTinyBook/Z-Ant/contributors). Thank you for helping shape the future of tinyML!

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file in the repository.

---

<div align="center">

**Join us in revolutionizing AI on edge devices! 🚀**

[GitHub](https://github.com/ZIGTinyBook/Z-Ant) • [Documentation](docs/) • [Examples](examples/) • [Community](https://github.com/ZIGTinyBook/Z-Ant/discussions)

</div>
