# STM32N6 Accelerator Integration – Detailed Notes

## Build system updates
- Added `configureStm32n6Support` in `build.zig` to register the STM32N6 C sources, propagate feature macros, and inject optional CMSIS and Ethos-U include paths whenever `-Dstm32n6_accel` is enabled.【F:build.zig†L1-L41】【F:build.zig†L58-L78】
- Exposed new build options so end users can opt into the accelerator (`-Dstm32n6_accel`), force the host fallback (`-Dstm32n6_force_native`), and independently toggle Helium (`-Dstm32n6_use_cmsis`) or Ethos-U (`-Dstm32n6_use_ethos`) paths, along with optional include roots for both SDKs.【F:build.zig†L43-L78】
- Ensured every build artifact that pulls in the tensor module links against the STM32N6 shim and its dependencies whenever the accelerator flag is set, so tests, libraries, and generated binaries all see a consistent view of the hardware hooks.【F:build.zig†L108-L147】
- Updated `.gitignore` and the dependency manifest to accommodate local toolchains and optional SDK checkouts used by the accelerator helpers.【F:.gitignore†L1-L43】【F:.gitignore†L53-L56】【F:build.zig.zon†L1-L7】

## Accelerator dispatch layer
- Introduced `src/Core/Tensor/Accelerators/common.zig` to share convolution metadata (stride, dilation, padding, grouping) between Zig and C components.【F:src/Core/Tensor/Accelerators/common.zig†L1-L16】
- Replaced ad-hoc conditional compilation with a dedicated accelerator facade (`mod.zig`) that selects the STM32N6 backend when enabled and exposes introspection helpers used by tests.【F:src/Core/Tensor/Accelerators/mod.zig†L1-L41】
- Added a `null_accelerator` backend so builds without STM32N6 support keep compiling while reporting that no accelerated path ran.【F:src/Core/Tensor/Accelerators/null_accelerator.zig†L1-L23】
- Implemented `stm32n6.zig` to bridge Zig tensors to the C shim, enforce shape/type guards, and route calls through Ethos-U, Helium/CMSIS, or reference implementations depending on the build flags and runtime availability.【F:src/Core/Tensor/Accelerators/stm32n6.zig†L1-L116】

## Convolution bridge implementation
- Created `conv_kernels.h` as the shared header describing the exported C entry points and instrumentation APIs consumed by both Zig and the regression harness.【F:src/Core/Tensor/Accelerators/stm32n6/conv_kernels.h†L1-L47】
- Implemented `conv_f32.c`, which now contains:
  - A portable reference convolution that mirrors the previous Zig loop so numerical behaviour stays identical in all fallbacks.【F:src/Core/Tensor/Accelerators/stm32n6/conv_f32.c†L157-L248】
  - A Helium/CMSIS pathway that repacks tensors into the NHWC/q7 layout expected by `arm_convolve_s8`, quantizes activations and biases, sizes the CMSIS workspace dynamically, and marks the CMSIS flag so tests can verify the accelerated code ran.【F:src/Core/Tensor/Accelerators/stm32n6/conv_f32.c†L27-L155】【F:src/Core/Tensor/Accelerators/stm32n6/conv_f32.c†L249-L305】
  - Shared helpers for dot products, workspace management, and instrumentation used by both the reference and accelerated paths.【F:src/Core/Tensor/Accelerators/stm32n6/conv_f32.c†L17-L25】【F:src/Core/Tensor/Accelerators/stm32n6/conv_f32.c†L307-L339】
- Added `ethos_stub.c` so Ethos-U builds can link cleanly even without the Arm driver; the stub raises a test flag and forwards to the reference kernel until a real Ethos implementation is provided.【F:src/Core/Tensor/Accelerators/stm32n6/ethos_stub.c†L1-L49】

## Tensor math integration
- Updated `op_convolution.zig` to call the accelerator facade before executing the Zig fallback loop, passing along pre-validated convolution parameters and bias data. The portable code path remains untouched and runs when the accelerator declines to handle a case.【F:src/Core/Tensor/TensorMath/op_convolution.zig†L1-L214】

## Test coverage additions
- Added `test_op_convolution_stm32n6.zig`, a Zig unit test that runs only when both `-Dstm32n6_accel` and `-Dstm32n6_force_native` are set. It exercises the accelerator shim, checks the Ethos/CMSIS instrumentation flags, and compares the output tensor against the reference values element by element.【F:tests/Core/Tensor/TensorMath/test_op_convolution_stm32n6.zig†L1-L79】
- Registered the new test module in the tensor math aggregation suite so it runs with the rest of the operator coverage when the accelerator flags are active.【F:tests/Core/Tensor/TensorMath/test_tensor_math.zig†L1-L18】
- Authored `scripts/test_stm32n6_conv.py`, which builds three shared-library variants (reference, CMSIS/Helium, and Ethos) using the host C compiler, runs them via `ctypes`, and asserts both the numerical outputs and the execution flags for each path.【F:scripts/test_stm32n6_conv.py†L1-L120】【F:scripts/test_stm32n6_conv.py†L122-L210】
- Added a full QEMU regression harness (`scripts/test_stm32n6_qemu.py`) that cross-compiles bare-metal firmware for all three paths using Zig, arm-none-eabi-gcc, or a host GCC fallback, auto-detects CMSIS headers and the `arm_convolve_s8` source, and drives `qemu-system-arm` (or the bundled stub) until the semihosted PASS marker is observed.【F:scripts/test_stm32n6_qemu.py†L1-L210】【F:scripts/test_stm32n6_qemu.py†L212-L309】【F:scripts/test_stm32n6_qemu.py†L311-L402】
- Implemented the bare-metal firmware (`tests/stm32n6_qemu`) that invokes each convolution entry point, validates outputs, and reports results through semihosting, plus minimalist runtime, libc stubs, and linker script so the binary runs on Cortex-M55 or as a host fallback when `STM32N6_HOST` is defined.【F:tests/stm32n6_qemu/main.c†L1-L104】【F:tests/stm32n6_qemu/runtime.c†L1-L44】【F:tests/stm32n6_qemu/semihost.c†L1-L58】【F:tests/stm32n6_qemu/support.c†L1-L76】【F:tests/stm32n6_qemu/stm32n6.ld†L1-L43】

## CMSIS stub fixtures
- Added lightweight CMSIS-DSP/NN headers and a portable `arm_convolve_s8` reference implementation under `tests/fixtures/cmsis_stub` so the Helium path can be tested offline without cloning the full CMSIS tree.【F:tests/fixtures/cmsis_stub/arm_math.h†L1-L15】【F:tests/fixtures/cmsis_stub/arm_nnfunctions.h†L1-L46】【F:tests/fixtures/cmsis_stub/arm_convolve_s8.c†L1-L83】

## Tooling and dependency helpers
- Introduced `scripts/fetch_cmsis_nn.sh` and `scripts/fetch_ethos_u_driver.sh` to download, update, or install the CMSIS-NN and Ethos-U driver sources from local archives or mirrored repositories, simplifying setup in offline environments.【F:scripts/fetch_cmsis_nn.sh†L1-L70】【F:scripts/fetch_ethos_u_driver.sh†L1-L70】
- Added `scripts/install_zig.sh` to fetch a project-local Zig 0.14 toolchain, with overrides for alternate mirrors and archive staging directories.【F:scripts/install_zig.sh†L1-L49】
- Added `scripts/install_qemu.sh` to install `qemu-system-arm` via the host package manager (apt/dnf/yum/pacman/zypper/brew) so the STM32N6 QEMU regression can run on real emulated hardware without manual setup.【F:scripts/install_qemu.sh†L1-L85】
- Included a small Python-based `qemu-system-arm` stub so the regression harness can still execute host binaries when a real QEMU installation is unavailable.【F:scripts/qemu-system-arm†L1-L35】

## Documentation refresh
- Expanded the root `README.md` with the new STM32N6 build flags, testing commands, dependency-fetch helpers, and instructions for installing Zig locally in constrained environments.【F:README.md†L22-L125】【F:README.md†L135-L189】
- Documented the QEMU installation helper and expanded the STM32N6 accelerator guide with step-by-step environment preparation instructions, covering Zig, CMSIS, QEMU, and cross-compilers.【F:README.md†L30-L79】【F:README.md†L101-L115】【F:docs/accelerators/stm32n6.md†L70-L116】
- Authored `docs/accelerators/stm32n6.md` to capture the accelerator roadmap, configuration flags, testing strategy, and external SDK requirements.【F:docs/accelerators/stm32n6.md†L1-L123】
- Documented how the `third_party` directory hosts optional CMSIS-NN and Ethos-U driver archives that the helper scripts install on demand.【F:third_party/README.md†L1-L23】

## Third-party and repository layout changes
- Adjusted ignore rules and documentation so optional SDKs installed under `third_party/` do not pollute source control.【F:.gitignore†L33-L41】【F:third_party/README.md†L1-L23】

