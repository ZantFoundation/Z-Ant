# Z-Ant

<div align="left">
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-tests.yml/badge.svg" alt="Zig Tests" />
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zant-benchmarks.yml/badge.svg" alt="Zig Benchamrk Tests" />
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-codegen-tests.yml/badge.svg" alt="Zig Codegen Tests" />
</div>

<!-- BEER_TIMINGS_START -->
Beer model timing (QEMU, Cortex-M55):

- Reference: 944.38 ms
- CMSIS-NN: 836.23 ms
- Improvement: 108.15 ms (11.5%)
<!-- BEER_TIMINGS_END -->









![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)

## 🛠️ CI/CD

- `zig-tests` – regression suite covering the core runtime.
- `zig-codegen-tests` – validates generated operators and glue code.
- `zant-benchmarks` – runs the Beer end-to-end benchmark and refreshes the metrics below.

### 📈 Performance Snapshot


## Project Overview

## ✨ Why Z-Ant?

- **⚡ Microsecond** inference on ARM Cortex-M, RISC-V, x86
- **📦 Zero dependencies** - single static library deployment
- **🎯 ONNX native** - direct model deployment from ONNX
- **🔧 30+ operators** - comprehensive neural network support
- **📷 Built-in image processing** - JPEG decode + preprocessing
- **🧠 Smart optimization** - quantization, pruning, memory efficiency

## Use Cases

- **🏭 Edge AI**: Real-time anomaly detection, predictive maintenance
- **🤖 IoT & Autonomous Systems**: Lightweight AI models for drones, robots, vehicles, IoT devices
- **📱 Mobile Applications**: On-device inference for privacy-preserving AI
- **🏥 Medical Devices**: Real-time health monitoring and diagnostics
- **🎮 Gaming**: AI-powered gameplay enhancement on embedded systems

---

## 🚀 Quick Start
Prerequisites

- [Zig 0.14.x](https://ziglang.org/learn/getting-started/) *(run `./scripts/install_zig.sh` to fetch a local copy; set `ZIG_DOWNLOAD_URL=file:///absolute/path/to/zig-linux-x86_64-0.14.0.tar.xz` when working from a pre-downloaded archive)*
- `qemu-system-arm` 7.2+ *(install via `./scripts/install_qemu.sh` or your platform package manager when running the STM32 N6 QEMU harness)*

### Get Started in 2 Minutes
```bash
# Clone and verify installation
git clone https://github.com/ZantFoundation/Z-Ant.git
cd Z-Ant

# Install Zig 0.14 locally (optional if you already have it)
./scripts/install_zig.sh
# (use ZIG_DOWNLOAD_URL or ZIG_DOWNLOAD_BASE to point at a mirror if needed)
export PATH="$(pwd)/.zig-toolchain/current:$PATH"

# - put your onnx model inside /datasets/models in a folder with the same of the model to to have: /datasets/models/my_model/my_model.onnx

# - simplify and prepare the model for zant inference engine
./zant input_setter --path /datasets/models/my_model/my_model.onnx --shape "your,model,sha,pe"

# - Generate test data
./zant user_tests_gen --model my_model

# --- GENERATING THE Single Node lib and test it ---
#For a N nodes model it creates N onnx models, one for each node with respective tests.
./zant onnx_extract --path /datasets/models/my_model/my_model.onnx

#generate libs for extracted nodes
zig build extractor-gen -Dmodel="my_model"

#test extracted nodes
zig build extractor-test -Dmodel="my_model" 

# --- GENERATING THE LIBRARY and TESTS ---
# Generate code for a specific model
zig build lib-gen -Dmodel="my_model" -Denable_user_tests [-Ddynamic -Ddo_export -Dlog -Dcomm ... ]

# Test the generated code
zig build lib-test -Dmodel="my_model" -Denable_user_tests [-Ddynamic -Ddo_export -Dlog -Dcomm ... ]

# Build the static library
zig build lib -Dmodel="my_model" [-Dtarget=... -Dcpu=...]

```

## 📖 Essential Commands  

**IMPORTANT**: see [ZANT CLI](docs/ZANT_CLI.md) for a better understanding and more details!

### Core Workflow
| Command | What it does |
|---------|--------------|
| `zig build test` | Verify everything works |
| `zig build codegen -Dmodel=<name>` | Generate code from ONNX model |
| `zig build lib -Dmodel=<name>` | Build deployable static library |
| `zig build test-generated-lib -Dmodel=<name>` | Test your generated code |

### Target Platforms
| Platform | Target Flag | CPU Examples |
|----------|-------------|--------------|
| **ARM Cortex-M** | `-Dtarget=thumb-freestanding` | `-Dcpu=cortex_m33`, `-Dcpu=cortex_m4`, `-Dcpu=cortex_m55` |
| **RISC-V** | `-Dtarget=riscv32-freestanding` | `-Dcpu=generic_rv32` |
| **x86/Native** | `-Dtarget=native` | (auto-detected) |

### Key Options
| Option | Description | Example |
|--------|-------------|---------|
| `-Dmodel=<name>` | Your model name | `-Dmodel=my_classifier` |
| `-Dmodel_path=<path>` | Custom ONNX file | `-Dmodel_path=models/custom.onnx` |
| `-Dlog=true` | Enable detailed logging | `-Dlog=true` |
| `-Dcomm=true` | Add comments to generated code | `-Dcomm=true` |
| `-Dstm32n6_accel=true` | Enable STM32 N6 accelerator dispatch layer | `-Dstm32n6_accel=true` |
| `-Dstm32n6_cmsis_path=/abs/path` | Optional CMSIS include root used when the accelerator flag is set | `-Dstm32n6_cmsis_path="/opt/CMSIS_6/Source"` |
| `-Dstm32n6_use_cmsis=true` | Use CMSIS Helium helpers (requires CMSIS-DSP headers or `third_party/CMSIS-NN`) | `zig build test -Dstm32n6_accel=true -Dstm32n6_use_cmsis=true` |
| `-Dstm32n6_use_ethos=true` | Enable Ethos-U execution path (requires Ethos-U driver headers) | `zig build test -Dstm32n6_accel=true -Dstm32n6_use_ethos=true` |
| `-Dstm32n6_force_native=true` | Force the STM32 N6 accelerator shim to run on the host (useful for smoke tests) | `zig build test -Dstm32n6_accel=true -Dstm32n6_force_native=true` |

### Optional SDK downloads
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

### STM32 N6 accelerator testing

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

## 🔧 ONNX Tools (Python Helpers)

Z-Ant includes Python scripts for ONNX model preparation:

```bash
# Prepare your model: set input shapes and infer all tensor shapes
./zant input_setter --path model.onnx --shape 1,3,224,224

# Generate test data for validation
./zant user_tests_gen --model model.onnx --iterations 10

# Create operator test models
./zant onnx_gen --op Conv --iterations 5
```

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
├── src/                    # Core source code
│   ├── Core/              # Neural network core functionality
│   ├── CodeGen/           # Code generation engine
│   ├── ImageToTensor/     # Image preprocessing pipeline
│   ├── onnx/              # ONNX model parsing
│   └── Utils/             # Utilities and helpers
├── tests/                 # Comprehensive test suite
├── datasets/              # Sample models and test data
├── generated/             # Generated code output
├── examples/              # Arduino and microcontroller examples
└── docs/                  # Documentation and guides
```

---

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
- Follow our [Code of Conduct](docs/CODE_OF_CONDUCT.md)
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

 
