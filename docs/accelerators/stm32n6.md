# STM32 N6 Accelerator Support Plan

This document captures the roadmap for enabling hardware acceleration on STM32 N6 devices within Z-Ant. The STM32 N6 combines an Arm Cortex-M55 core with Helium (M-Profile Vector Extension) and integrates the Arm Ethos-U NPU. Our goal is to progressively expose these accelerators to Z-Ant generated inference libraries while keeping the default portable path available.

## 1. Platform Overview

| Component | Notes |
|-----------|-------|
| **Cortex-M55 + Helium** | Provides MVE vector units (SIMD for int8/f16/f32) suited to convolution, matrix multiply and activation kernels. |
| **Ethos-U NPU** | Dedicated ML accelerator with its own command stream; accessed through Arm Ethos-U driver stack. |
| **CMSIS-NN / CMSIS-DSP** | Reference optimized kernels for Cortex-M targets, including MVE-optimized implementations. |

## 2. Integration Strategy

1. **Configurable build flag** – expose `-Dstm32n6_accel=true` in `zig build` so users can opt-in when targeting STM32 N6 firmware.
2. **Accelerator abstraction** – add a lightweight dispatch layer that lets tensor operators query whether a hardware-specific kernel is available, falling back to the existing pure Zig implementation when not.
3. **Helium friendly kernels** – surface hooks for convolution (and later GEMM, pooling) that can call into optimized routines. A C bridge compiled via `zig cc` is used so we can link either our stub kernels or CMSIS sources.
4. **External SDK linkage** – allow passing `-Dstm32n6_cmsis_path=/path/to/CMSIS` to pull in the CMSIS-NN sources and headers; the build script wires these into every artifact that needs the accelerated kernels.
5. **Ethos-U integration** – future phase: provide a command-stream generator for Ethos-U (via Arm’s driver). The dispatch layer is designed so we can attach an Ethos-U execution path for entire subgraphs or keep using Helium for individual operators.

## 3. Repository Changes Implemented

* `src/Core/Tensor/Accelerators/` contains the new dispatch layer. When `stm32n6_accel` is enabled, the STM32 N6 backend is compiled; otherwise a null backend is used.
* `src/Core/Tensor/TensorMath/op_convolution.zig` now asks the accelerator backend whether it can execute the convolution. If the backend returns `false` the existing Zig loops run exactly as before.
* `src/Core/Tensor/Accelerators/stm32n6/conv_f32.c` is a reference C implementation that mirrors the current Zig convolution logic and exposes hooks for both the CMSIS Helium path and the portable fallback.
* `src/Core/Tensor/Accelerators/stm32n6/ethos_stub.c` implements the Ethos-U dispatch hook. When the real driver is not present it forwards to the reference kernel while keeping per-path instrumentation for tests.
* `build.zig` wires the new option into every target (tests, executables and generated libraries) and pulls in the STM32 specific sources and include paths when the flag is active. Additional build flags (`stm32n6_use_cmsis`, `stm32n6_use_ethos`) toggle the Helium and Ethos execution paths independently.
* `scripts/test_stm32n6_conv.py` now compiles three variants (reference, CMSIS and Ethos) so the bridging logic can be validated even without the Zig compiler.

## 4. How to Use the STM32 N6 Backend

```bash
zig build lib-gen -Dmodel=my_model -Dstm32n6_accel=true \
  -Dstm32n6_use_cmsis=true -Dstm32n6_cmsis_path="/absolute/path/to/CMSIS_6/Source"
zig build lib -Dmodel=my_model -Dstm32n6_accel=true \
  -Dstm32n6_use_cmsis=true -Dstm32n6_use_ethos=true \
  -Dstm32n6_cmsis_path="/absolute/path/to/CMSIS_6/Source" \
  -Dstm32n6_ethos_path="/absolute/path/to/ethos-u-core-driver"
```

* The `stm32n6_cmsis_path` flag is optional; if omitted the build uses the included reference kernel. Enable `stm32n6_use_cmsis=true` to compile against the CMSIS Helium path when the headers are available.
* Run `./scripts/fetch_cmsis_nn.sh` to download Arm's CMSIS-NN sources into `third_party/CMSIS-NN` and make them the default
  include path for accelerated builds (set `CMSIS_NN_REPO`/`CMSIS_NN_REF` to use mirrors or specific releases, or
  `CMSIS_NN_ARCHIVE=/absolute/path/to/CMSIS-NN-main.zip` to install from a local archive when network access is restricted).
* Run `./scripts/fetch_ethos_u_driver.sh` to mirror the Arm Ethos-U core driver into `third_party/ethos-u-core-driver` when the Ethos path is required (override `ETHOS_U_REPO`/`ETHOS_U_REF`/`ETHOS_U_ARCHIVE` for mirrors or offline installs).
* When providing the CMSIS path, both the DSP and NN include directories should be present under the supplied folder.
* Additional CMSIS or board support sources can be appended by extending `build.zig` – the helper already has a single place where the list of source files can be augmented.

### Host smoke testing

* Pass `-Dstm32n6_force_native=true` together with `-Dstm32n6_accel=true` to force the accelerator shim to run on the build
  host. This bypasses the Thumb-only guard so you can execute the accelerator unit test suite locally:

  ```bash
  zig build test -Dstm32n6_accel=true -Dstm32n6_force_native=true
  ```

* When the Zig toolchain is not available you can still exercise the C shim directly with clang. The helper script builds
  reference, CMSIS and Ethos variants and checks the run-time instrumentation flags:

  ```bash
  ./scripts/test_stm32n6_conv.py
  ```

### QEMU regression harness

* The repository ships with a bare-metal firmware harness that validates the reference, CMSIS/Helium and Ethos entry points and
  asserts that the instrumentation flags reflect which path executed. It uses Arm semihosting to report pass/fail states so the
  test can run unattended under QEMU.

* Requirements:

  * Either Zig 0.14 (so `zig cc` can target `thumbv8m.main-none-eabi`) or the GNU Arm Embedded toolchain (`arm-none-eabi-gcc`).
    When neither is available the helper falls back to the system `gcc` and defines `STM32N6_HOST=1` so the harness builds as a
    normal Linux binary.
* `qemu-system-arm` with support for the `mps3-an547` board (Cortex-M55). Install it from your distribution, run
  `./scripts/install_qemu.sh` on common Linux/macOS setups, or point the regression helper at a locally extracted archive.
  As a last resort the repository provides `scripts/qemu-system-arm`, a small wrapper that simply runs the host binary so
  the regression still executes in restricted sandboxes.
* CMSIS headers providing both `arm_math.h` and `arm_nnfunctions.h`, plus the Helium convolution implementation
  (`arm_convolve_s8.c`). The script falls back to the stub fixtures in `tests/fixtures/cmsis_stub` when a full CMSIS
  checkout is not present.

#### Preparing your environment

1. **Install Zig 0.14** – use `./scripts/install_zig.sh` to fetch a local toolchain or point the build at an existing
   installation.
2. **Mirror CMSIS-NN (optional)** – invoke `./scripts/fetch_cmsis_nn.sh` to populate `third_party/CMSIS-NN` so the Helium
   path can link the official kernels instead of the stub fixtures.
3. **Install `qemu-system-arm`** – run `./scripts/install_qemu.sh` (requires administrator privileges) or install the
   package manually via your distribution/Homebrew. When offline, download a suitable package/archive first and expose the
   unpacked binary via the `--qemu` flag.
4. **Provide a cross compiler (optional)** – install the GNU Arm Embedded toolchain or point `--zig` at a Zig 0.14 binary
   if you prefer to produce a real Thumb bare-metal image. The helper automatically falls back to the host GCC when no
   cross compiler is present.

* Example invocation (using the stubbed CMSIS sources and auto-detected toolchain/QEMU binaries):

  ```bash
  ./scripts/test_stm32n6_qemu.py
  ```

* When running in an offline environment, use the overrides to point at locally unpacked archives:

  ```bash
  ./scripts/test_stm32n6_qemu.py \
    --zig /path/to/zig \
    --qemu /path/to/qemu-system-arm \
    --cmsis-include /path/to/CMSIS/DSP/Include \
    --cmsis-nn-include /path/to/CMSIS/NN/Include \
    --cmsis-convolve /path/to/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
  ```

## 5. Next Steps

1. Replace the reference convolution with the CMSIS-NN Helium kernel (`arm_convolve_s8`, `arm_convolve_fast_s16`, etc.), guarding each dtype with `@hasDecl` checks.
2. Add support for other hot operators (depthwise convolution, matrix multiply, pooling) through the same dispatch layer.
3. Integrate Ethos-U: detect Ethos-U capable builds, generate intermediate command buffers and fall back to Helium for unsupported operators.
4. Provide regression tests that run the accelerated kernels on QEMU/FVP, comparing them against the Zig implementation for multiple tensor shapes and quantization schemes.
5. Document board-level integration (clock setup, DMA, memory placement) in a dedicated guide once the first firmware example is ready.

This staged approach keeps the core code generator stable while enabling incremental adoption of STM32 N6 accelerators.
