# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-07-18

Release cut from `feature` to `main`: ~390 commits / 78 merged PRs since `v1.0.0`.

### Breaking Changes

- Removed the monolithic `zant` facade module (`src/zant.zig`). The build now exposes four
  independent Zig build modules — `zant_utils`, `IR_zant`, `codegen`, and `TensorToImage` — each
  with an explicit, minimal dependency edge (no circular deps). Code that imported `@import("zant")`
  must import the specific module it needs instead.
- Directory layout restructured: `src/Core/`, `src/Utils/`, `src/IR_zant/` moved and consolidated
  under `src/utils/` and `src/codegen/IR_zant/`; `src/codegen/cg_v1/` and `src/codegen/cg_v2/` were
  merged/renamed (e.g. `src/codegen/cg_v1/predict/emit.zig` → `src/codegen/predict/emit.zig`).
  `tests/Core/`, `tests/Utils/`, `tests/IR_graph/`, `tests/Onnx/`, `tests/test_lib.zig` were replaced
  by per-module test aggregators colocated with each module (e.g. `src/utils/utils_tests.zig`,
  `src/codegen/IR_zant_tests.zig`).
- Removed the STM32N6/CMSIS-NN/Ethos-U accelerator scripts and docs
  (`scripts/fetch_cmsis_nn.sh`, `scripts/fetch_ethos_u_driver.sh`, `scripts/test_stm32n6_*`,
  `docs/accelerators/*`) and the `beer-regression` CI workflow.

### Added

- **GAF/MTF time-series-to-image encoding** — new `TensorToImage` module: Gramian Angular
  Summation/Difference Field (`gasf.zig`, `gadf.zig`), Markov Transition Field (`mtf/`), a
  compound RGB (GASF+GADF+MTF) encoder for CNN input (`compound.zig`), a colormap module
  (viridis/jet/grayscale), and a BMP writer (`matrixToBmp.zig`). Includes a `zig build gaf-demo`
  example and a Nicla ECG classifier Arduino example (`examples/Nicla-ecg/`).
- **Static memory planning** — compile-time buffer-reuse planning for statically-allocated
  generated code (`-Ddynamic=false -Dstatic_planning=<heuristic>`), with heuristics
  `pressure_then_size`, `pressure_then_liveness`, `liveness_first`, `size_first`, `first_step`
  (and their inverse variants), optional branch-and-bound search (`-Dforce_bnb`), a visualisation
  script (`scripts/visualise_static_memory_allocation.py`), and documentation
  (`docs/static_memory_planning.md`).
- **New operators**: `Abs`, plus improvements to `Pow` (mixed-precision and broadcast edge cases).
- **New fusion patterns**: `Conv+Sigmoid+Mul` (YOLO-style blocks), `Pad+Conv`,
  `Dequant+Clip+Quant`, `Dequant+Pad+Quant+QLinearConv`, `Quant+Dequant`, in addition to the
  existing `Conv+Relu` / `Conv+Clip` patterns.
- **NCHW/NHWC layout conversion** support in the tensor/codegen pipeline.
- **New codegen output targets**: Arduino `.ino` file generation (`-Dgen_ino`) and C `.h` header
  generation (`-Dgen_h`).
- New `docs/static_memory_planning.md`, `docs/scripts_for_static_memory_allocation.md`, and this
  `CHANGELOG.md`.

### Changed

- Migrated the project to **Zig 0.15.2** (from the previous toolchain version).
- Reworked `build.zig` and the `zantBuild/` module (`ZantModules`, `ZantOptions`,
  `ZantStepOptions`) to wire up modules and build steps explicitly per-module instead of through
  the old facade.
- Reorganized CI: `zig-tests`, `zig-codegen-tests`, and `zig-benchmarks` (renamed from
  `zant-benchmarks`) GitHub Actions workflows, with shared `setup-zig` / `setup-python` composite
  actions.
- ONNX operator identification refactored to a typed enum instead of ad-hoc string matching.

### Fixed

- Fixed a double-free (segfault) in generated dynamic-allocation code caused by an unnecessary
  `defer tensor.deinit()` on tensors already freed by the execution plan's `step.frees`
  (`src/codegen/predict/emit.zig`).
- Various tensor math, quantization, and codegen correctness fixes accumulated across the
  `core_refactor`, `static-memory-planning-*`, and `nchw-nhwc-conversion` branches (see individual
  PRs #450–#513 for details).

[2.0.0]: https://github.com/ZantFoundation/Z-Ant/compare/v1.0.0...main
