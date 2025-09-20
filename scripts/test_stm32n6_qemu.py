#!/usr/bin/env python3
"""Cross-compile the STM32 N6 convolution harness and run it under QEMU."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
HARNESS_COMMON_DIR = HARNESS_DIR / "common"
HARNESS_OPS_DIR = HARNESS_DIR / "ops"
CONV_HARNESS_DIR = HARNESS_OPS_DIR / "conv"
BEER_HARNESS_DIR = HARNESS_DIR / "beer"
CMSIS_STUB_DIR = REPO_ROOT / "tests/fixtures/cmsis_stub"
LINKER_SCRIPT = HARNESS_DIR / "stm32n6.ld"
BEER_LIB_PATH = REPO_ROOT / "zig-out" / "beer" / "libzant.a"
BEER_GENERATED_DIR = REPO_ROOT / "generated" / "beer"
COMMON_BARE_METAL_SOURCES = (
    HARNESS_COMMON_DIR / "runtime.c",
    HARNESS_COMMON_DIR / "semihost_arm.c",
    HARNESS_COMMON_DIR / "support.c",
)

COMMON_HOST_SOURCES = (
    HARNESS_COMMON_DIR / "semihost.c",
    HARNESS_COMMON_DIR / "support.c",
)


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


class ToolchainError(RuntimeError):
    pass


class Toolchain:
    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def default_macros(self) -> Sequence[str]:
        return ()

    def extra_include_dirs(self) -> Sequence[Path]:
        return ()


class ZigToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "cc",
            "-Dtarget=thumb-freestanding",
            "-Dcpu=cortex_m55",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-Wl,-T," + str(LINKER_SCRIPT),
            "-O2",
            "-g",
            "-nostdlib",
            "-o",
            str(output),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        for include in (*self.extra_include_dirs(), *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"zig cc ({self.exe})"


class ArmGccToolchain(Toolchain):
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.cc = shutil.which(f"{prefix}-gcc")
        if self.cc is None:
            raise ToolchainError(f"unable to locate {prefix}-gcc in PATH")

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            self.cc,
            "-mcpu=cortex-m55",
            "-mthumb",
            "-mfloat-abi=soft",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-O2",
            "-g",
            "-nostdlib",
            "-Wl,-T",
            str(LINKER_SCRIPT),
            "-Wl,--gc-sections",
            "-Wl,--nmagic",
            "-Wl,-Map=" + str(output.with_suffix(".map")),
            "-o",
            str(output),
        ]
        for include in (*self.extra_include_dirs(), *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        cmd.extend(["-Wl,--start-group", "-lgcc", "-Wl,--end-group"])
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"{self.prefix}-gcc ({self.cc})"


class HostGccToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "-std=c11",
            "-O2",
            "-g",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-o",
            str(output),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        for include in (*self.extra_include_dirs(), *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        cmd.append("-lm")
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"host gcc ({self.exe})"

    def default_macros(self) -> Sequence[str]:
        return ("STM32N6_HOST=1",)


class ClangToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "-Dtarget=thumb-freestanding",
            "-Dcpu=cortex_m55",
            "-mthumb",
            "-mfpu=fpv5-sp-d16",
            "-mfloat-abi=hard",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-fdata-sections",
            "-ffunction-sections",
            "-O2",
            "-g",
            "-fuse-ld=lld",
            "-nostdlib",
            "-o",
            str(output),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        for include in (*self.extra_include_dirs(), *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"clang ({self.exe})"


class BuildFlavor(Enum):
    BARE_METAL = "bare_metal"
    HOST = "host"
    BEER = "beer"


@dataclass(frozen=True)
class CaseTemplate:
    name: str
    macros: tuple[str, ...] = ()
    needs_cmsis: bool = False
    needs_ethos: bool = False


@dataclass(frozen=True)
class FirmwareCase:
    template: CaseTemplate
    name: str
    macros: tuple[str, ...]
    extra_sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    success_marker: str | None


@dataclass(frozen=True)
class CaseTiming:
    case: FirmwareCase
    durations: list[float]


class CmsisBundle:
    def __init__(self, dsp_include: Path, nn_include: Path, convolve_source: Path):
        self.dsp_include = dsp_include
        self.nn_include = nn_include
        self.convolve_source = convolve_source

    def include_dirs(self) -> tuple[Path, ...]:
        dirs: list[Path] = [self.dsp_include]
        if self.nn_include != self.dsp_include:
            dirs.append(self.nn_include)
        if self.dsp_include != CMSIS_STUB_DIR:
            cmsis_core = REPO_ROOT / "third_party" / "CMSIS_5" / "CMSIS" / "Core" / "Include"
            if cmsis_core.exists():
                dirs.append(cmsis_core)
        return tuple(_dedupe_paths(dirs))

    def cmsis_sources(self) -> tuple[Path, ...]:
        return tuple(
            _dedupe_paths(
                [
                    *get_cmsis_sources(self.convolve_source, self.nn_include),
                    *get_cmsis_dsp_sources(self.dsp_include),
                ]
            )
        )


@dataclass(frozen=True)
class ToolchainSelection:
    toolchain: Toolchain
    flavor: BuildFlavor


class FirmwareOperation:
    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        elf_prefix: str,
        success_marker_template: str | None,
        base_sources: dict[BuildFlavor, Sequence[Path]],
        case_templates: Sequence[CaseTemplate],
        default_includes: Sequence[Path] = (),
        summary_title: str | None = None,
        baseline_case: str | None = None,
        delta_pairs: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.elf_prefix = elf_prefix
        self.success_marker_template = success_marker_template
        self._base_sources = dict(base_sources)
        self.case_templates = tuple(case_templates)
        self._default_includes = tuple(_dedupe_paths(default_includes))
        self.summary_title = summary_title or f"{self.display_name} timing summary:"
        self.baseline_case = baseline_case
        self.delta_pairs = tuple(delta_pairs)

    def base_sources(self, flavor: BuildFlavor) -> Sequence[Path]:
        try:
            return self._base_sources[flavor]
        except KeyError as exc:
            raise ToolchainError(
                f"operation '{self.name}' does not support {flavor.value} builds"
            ) from exc

    def default_include_dirs(self) -> tuple[Path, ...]:
        return self._default_includes

    def case_name(self, template: CaseTemplate) -> str:
        return template.name

    def elf_name(self, case_name: str) -> str:
        return f"{self.elf_prefix}_{case_name}.elf"

    def format_success_marker(self, case_name: str) -> str | None:
        if self.success_marker_template is None:
            return None
        return self.success_marker_template.format(case=case_name, operation=self.name)

    def case_include_dirs(self, cmsis: CmsisBundle) -> tuple[Path, ...]:
        return cmsis.include_dirs()

    def case_extra_sources(
        self,
        template: CaseTemplate,
        cmsis: CmsisBundle,
        cmsis_sources: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        if template.needs_cmsis or template.needs_ethos:
            return cmsis_sources
        return ()

    def build_cases(self, cmsis: CmsisBundle) -> list[FirmwareCase]:
        include_dirs = self.case_include_dirs(cmsis)
        cmsis_sources = cmsis.cmsis_sources()
        cases: list[FirmwareCase] = []
        for template in self.case_templates:
            case_name = self.case_name(template)
            cases.append(
                FirmwareCase(
                    template=template,
                    name=case_name,
                    macros=template.macros,
                    extra_sources=self.case_extra_sources(template, cmsis, cmsis_sources),
                    include_dirs=include_dirs,
                    success_marker=self.format_success_marker(case_name),
                )
            )
        return cases

    def print_summary(self, case_timings: Sequence[CaseTiming]) -> None:
        print_timing_summary(
            self.summary_title,
            case_timings,
            baseline=self.baseline_case,
            delta_pairs=self.delta_pairs,
        )


def _build_operation_sources(
    *,
    operation_sources: Sequence[Path],
    shared_sources: Sequence[Path] = (),
    bare_metal_only: Sequence[Path] = (),
    host_only: Sequence[Path] = (),
    include_host: bool = True,
) -> dict[BuildFlavor, tuple[Path, ...]]:
    bare_paths = _dedupe_paths(
        (*COMMON_BARE_METAL_SOURCES, *operation_sources, *shared_sources, *bare_metal_only)
    )
    sources: dict[BuildFlavor, tuple[Path, ...]] = {
        BuildFlavor.BARE_METAL: tuple(bare_paths)
    }
    if include_host:
        host_paths = _dedupe_paths(
            (*COMMON_HOST_SOURCES, *operation_sources, *shared_sources, *host_only)
        )
        sources[BuildFlavor.HOST] = tuple(host_paths)
    return sources


class ConvolutionOperation(FirmwareOperation):
    def __init__(self) -> None:
        super().__init__(
            name="stm32n6",
            display_name="STM32N6 QEMU",
            elf_prefix="stm32n6",
            success_marker_template="stm32n6 {case} PASS",
            base_sources=_build_operation_sources(
                operation_sources=(CONV_HARNESS_DIR / "main.c",),
                shared_sources=(
                    STM32_DIR / "stm32n6_common.c",
                    STM32_DIR / "conv_f32.c",
                    STM32_DIR / "ethos_stub.c",
                ),
            ),
            case_templates=(
                CaseTemplate("reference"),
                CaseTemplate(
                    "helium",
                    macros=("ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1"),
                    needs_cmsis=True,
                ),
                CaseTemplate(
                    "ethos",
                    macros=("ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1", "ZANT_HAS_ETHOS_U=1"),
                    needs_cmsis=True,
                    needs_ethos=True,
                ),
            ),
            default_includes=(
                HARNESS_DIR,
                HARNESS_COMMON_DIR,
                CONV_HARNESS_DIR,
                STM32_DIR,
            ),
            summary_title="Timing summary:",
            baseline_case="reference",
            delta_pairs=(("ethos", "helium"),),
        )


class BeerOperation(FirmwareOperation):
    def __init__(self, beer_lib: Path) -> None:
        super().__init__(
            name="beer",
            display_name="Beer model",
            elf_prefix="beer",
            success_marker_template="beer PASS",
            base_sources=_build_operation_sources(
                operation_sources=(BEER_HARNESS_DIR / "main.c",),
                shared_sources=(
                    STM32_DIR / "stm32n6_common.c",
                    STM32_DIR / "conv_f32.c",
                    STM32_DIR / "ethos_stub.c",
                ),
                include_host=False,
            ),
            case_templates=(
                CaseTemplate("reference"),
                CaseTemplate(
                    "helium",
                    macros=("ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1"),
                    needs_cmsis=True,
                ),
            ),
            default_includes=(
                HARNESS_DIR,
                HARNESS_COMMON_DIR,
                BEER_HARNESS_DIR,
                STM32_DIR,
                BEER_GENERATED_DIR,
            ),
            summary_title="Beer model timing summary:",
            baseline_case="beer_reference",
        )
        self.beer_lib = beer_lib

    def case_name(self, template: CaseTemplate) -> str:
        return f"beer_{template.name}"

    def case_include_dirs(self, cmsis: CmsisBundle) -> tuple[Path, ...]:
        dirs = list(cmsis.include_dirs())
        dirs.append(BEER_GENERATED_DIR)
        return tuple(_dedupe_paths(dirs))

    def case_extra_sources(
        self,
        template: CaseTemplate,
        cmsis: CmsisBundle,
        cmsis_sources: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        # Both cases link against the beer library; ensure CMSIS sources are present
        return tuple((*cmsis_sources, self.beer_lib))

    def format_success_marker(self, case_name: str) -> str | None:
        return "beer PASS"


_OPERATION_FACTORIES: dict[str, type[FirmwareOperation]] = {
    "conv": ConvolutionOperation,
}

_OPERATION_ALIASES: dict[str, str] = {
    "conv": "conv",
    "convolution": "conv",
    "stm32n6": "conv",
    "stm32n6-conv": "conv",
}

AVAILABLE_OPERATION_CHOICES = tuple(sorted(_OPERATION_ALIASES))


def canonicalize_operation_name(name: str) -> str:
    key = _OPERATION_ALIASES.get(name.lower(), name.lower())
    if key not in _OPERATION_FACTORIES:
        raise ToolchainError(f"unknown STM32N6 operation '{name}'")
    return key


def instantiate_operations(names: Sequence[str]) -> list[FirmwareOperation]:
    seen: set[str] = set()
    operations: list[FirmwareOperation] = []
    for name in names:
        canonical = canonicalize_operation_name(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        factory = _OPERATION_FACTORIES[canonical]
        operations.append(factory())
    return operations


class QemuManager:
    def __init__(self, executable: Path, *, timeout: float, verbose: bool) -> None:
        self.executable = executable
        self.timeout = timeout
        self.verbose = verbose

    def run(self, elf_path: Path, *, success_marker: str | None) -> subprocess.CompletedProcess[str]:
        return _run_qemu(
            self.executable,
            elf_path,
            verbose=self.verbose,
            success_marker=success_marker,
            timeout=self.timeout,
        )


class HarnessRunner:
    def __init__(
        self,
        *,
        operation: FirmwareOperation,
        selection: ToolchainSelection,
        qemu: QemuManager | None,
        build_dir: Path,
        repeat: int,
        verbose: bool,
    ) -> None:
        self.operation = operation
        self.selection = selection
        self.qemu = qemu
        self.build_dir = build_dir
        self.repeat = repeat
        self.verbose = verbose

    def run_cases(self, cases: Sequence[FirmwareCase]) -> list[CaseTiming]:
        if self.selection.flavor != BuildFlavor.HOST and self.qemu is None:
            raise ToolchainError(
                "QEMU is required for bare-metal STM32N6 toolchains; rerun with --qemu"
            )

        base_sources = self.operation.base_sources(self.selection.flavor)
        results: list[CaseTiming] = []
        for case in cases:
            elf_path = self.build_dir / self.operation.elf_name(case.name)
            print(f"\n[build] {case.name}")
            include_dirs = tuple(
                _dedupe_paths((*self.operation.default_include_dirs(), *case.include_dirs))
            )
            self.selection.toolchain.build(
                output=elf_path,
                base_sources=base_sources,
                macros=(*self.selection.toolchain.default_macros(), *case.macros),
                extra_sources=case.extra_sources,
                include_dirs=include_dirs,
            )
            durations: list[float] = []
            for iteration in range(self.repeat):
                if self.repeat > 1:
                    print(f"[run]   {case.name} ({iteration + 1}/{self.repeat})")
                else:
                    print(f"[run]   {case.name}")
                start = time.perf_counter()
                result = self._execute_case(case, elf_path)
                duration = time.perf_counter() - start
                durations.append(duration)
                if self.verbose:
                    sys.stdout.write(result.stdout)
                expected_marker = case.success_marker
                if result.returncode not in (0, 1):
                    raise RuntimeError(
                        f"QEMU exited with status {result.returncode} during {case.name} run:\n{result.stdout}"
                    )
                if expected_marker and expected_marker not in result.stdout:
                    raise RuntimeError(
                        f"Harness output missing PASS marker for {case.name}:\n{result.stdout}"
                    )
                if not self.verbose and expected_marker:
                    for line in result.stdout.splitlines():
                        if expected_marker in line:
                            print(line)
                            break
                print(f"✅ {case.name} completed in {duration * 1000.0:.2f} ms")
            results.append(CaseTiming(case=case, durations=durations))
        return results

    def _execute_case(
        self, case: FirmwareCase, elf_path: Path
    ) -> subprocess.CompletedProcess[str]:
        if self.selection.flavor == BuildFlavor.HOST:
            return subprocess.run(
                [str(elf_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        assert self.qemu is not None
        return self.qemu.run(elf_path, success_marker=case.success_marker)

def print_timing_summary(
    title: str,
    case_timings: Sequence[CaseTiming],
    *,
    baseline: str | None = None,
    delta_pairs: Sequence[tuple[str, str]] = (),
) -> None:
    if not case_timings:
        return

    print(f"\n{title}")
    means: dict[str, float] = {}
    for timing in case_timings:
        if not timing.durations:
            continue
        durations_ms = [value * 1000.0 for value in timing.durations]
        avg_ms = sum(durations_ms) / len(durations_ms)
        best_ms = min(durations_ms)
        means[timing.case.name] = avg_ms
        formatted = ", ".join(f"{value:.2f} ms" for value in durations_ms)
        print(
            f"  {timing.case.name}: mean {avg_ms:.2f} ms, min {best_ms:.2f} ms"
            f" over {len(durations_ms)} run(s) [{formatted}]"
        )

    if not means:
        return

    if baseline and baseline in means:
        baseline_avg = means[baseline]
        for name, avg in means.items():
            if name == baseline:
                continue
            delta_ms = avg - baseline_avg
            print(f"    Δ({name} - {baseline}): {delta_ms:.2f} ms")

    for lhs, rhs in delta_pairs:
        if lhs in means and rhs in means:
            delta_ms = means[lhs] - means[rhs]
            print(f"    Δ({lhs} - {rhs}): {delta_ms:.2f} ms")


def detect_zig(explicit: str | None) -> Path | None:
    candidates: Iterable[str] = ()
    if explicit:
        candidates = (explicit,)
    else:
        zig_env = os.environ.get("ZIG")
        if zig_env:
            candidates = (zig_env,)
        else:
            candidates = ("zig",)
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    return None


def detect_clang(explicit: str | None) -> Path | None:
    if explicit is None:
        return None
    resolved = shutil.which(explicit)
    if resolved:
        return Path(resolved)
    raise ToolchainError(f"unable to locate clang binary: {explicit}")


def detect_arm_gcc(prefix: str | None) -> str | None:
    prefixes: Iterable[str]
    if prefix:
        prefixes = (prefix,)
    else:
        env_prefix = os.environ.get("ARM_GNU_PREFIX")
        if env_prefix:
            prefixes = (env_prefix,)
        else:
            prefixes = ("arm-none-eabi",)
    for cand in prefixes:
        if shutil.which(f"{cand}-gcc"):
            return cand
    return None


def detect_host_gcc(explicit: str | None) -> Path | None:
    candidates: Iterable[str] = ()
    if explicit:
        candidates = (explicit,)
    else:
        env_candidate = os.environ.get("HOST_GCC")
        if env_candidate:
            candidates = (env_candidate,)
        else:
            candidates = ("gcc", "cc")
    for cand in candidates:
        resolved = shutil.which(cand)
        if resolved:
            return Path(resolved)
    return None


def detect_qemu(explicit: str | None) -> Path | None:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return Path(resolved)
        return None
    env_value = os.environ.get("QEMU_SYSTEM_ARM")
    if env_value and shutil.which(env_value):
        return Path(shutil.which(env_value))
    resolved = shutil.which("qemu-system-arm")
    if resolved:
        return Path(resolved)
    repo_stub = REPO_ROOT / "scripts" / "qemu-system-arm"
    if repo_stub.exists():
        return repo_stub
    return None


def find_arm_math_header(explicit: str | None) -> Path:
    candidates: Iterable[Path]
    if explicit:
        candidates = (Path(explicit),)
    else:
        env_candidate = os.environ.get("STM32N6_CMSIS_INCLUDE")
        search_roots: list[Path] = []
        if env_candidate:
            search_roots.append(Path(env_candidate))
        search_roots.extend(
            [
                REPO_ROOT / "third_party" / "CMSIS-DSP" / "Include",
                REPO_ROOT / "third_party" / "CMSIS_5" / "CMSIS" / "DSP" / "Include",
                REPO_ROOT / "third_party" / "CMSIS-NN" / "CMSIS" / "DSP" / "Include",
                CMSIS_STUB_DIR,
            ]
        )
        candidates = search_roots
    for candidate in candidates:
        header = candidate / "arm_math.h"
        if header.exists():
            return candidate
    raise ToolchainError(
        "unable to locate arm_math.h; pass --cmsis-include or set STM32N6_CMSIS_INCLUDE"
    )


def find_arm_nn_header(explicit: str | None, dsp_include: Path) -> Path:
    if explicit:
        candidate = Path(explicit)
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate
        raise ToolchainError(f"arm_nnfunctions.h not found under {candidate}")

    env_candidate = os.environ.get("STM32N6_CMSIS_NN_INCLUDE")
    if env_candidate:
        candidate = Path(env_candidate)
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate

    candidate_roots: list[Path] = []
    if dsp_include != CMSIS_STUB_DIR:
        candidate_roots.append(dsp_include.parent.parent / "NN" / "Include")
        candidate_roots.append(REPO_ROOT / "third_party" / "CMSIS-NN" / "Include")
    candidate_roots.append(CMSIS_STUB_DIR)

    for candidate in candidate_roots:
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate

    raise ToolchainError("unable to locate arm_nnfunctions.h; pass --cmsis-nn-include explicitly")


def find_arm_convolve_source(explicit: str | None, nn_include: Path) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise ToolchainError(f"arm_convolve_s8 source not found: {candidate}")
    if nn_include == CMSIS_STUB_DIR:
        return CMSIS_STUB_DIR / "arm_convolve_s8.c"

    source_root = nn_include.parent / "Source"
    matches = list(source_root.rglob("arm_convolve_s8.c")) if source_root.exists() else []
    if matches:
        return matches[0]

    stub = CMSIS_STUB_DIR / "arm_convolve_s8.c"
    if stub.exists():
        return stub

    raise ToolchainError("unable to find arm_convolve_s8.c; pass --cmsis-convolve explicitly")


def get_cmsis_sources(convolve_source: Path, nn_include: Path) -> tuple[Path, ...]:
    """Get all required CMSIS-NN source files"""
    sources = [convolve_source]

    # Add additional CMSIS-NN sources if using real CMSIS-NN (not stubs)
    if nn_include != CMSIS_STUB_DIR:
        cmsis_nn_source = nn_include.parent / "Source"
        if cmsis_nn_source.exists():
            additional_sources = [
                # Buffer size functions
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_get_buffer_sizes_s8.c",
                # Matrix multiplication kernels
                cmsis_nn_source / "ConvolutionFunctions" / "arm_nn_mat_mult_kernel_s8_s16.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
                # Support functions
                cmsis_nn_source / "NNSupportFunctions" / "arm_s8_to_s16_unordered_with_offset.c",
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_mat_mult_nt_t_s8.c",
                # Additional support functions that might be needed
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s8.c",
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_mat_mult_nt_t_s8_s32.c",
            ]
            # Only add files that exist
            for src in additional_sources:
                if src.exists():
                    sources.append(src)

    return tuple(sources)


def get_cmsis_dsp_sources(dsp_include: Path) -> tuple[Path, ...]:
    if dsp_include == CMSIS_STUB_DIR:
        return ()
    dsp_root = dsp_include.parent
    candidates = [
        dsp_root / "Source" / "BasicMathFunctions" / "arm_dot_prod_f32.c",
    ]
    return tuple(src for src in candidates if src.exists())


def _run_qemu(
    qemu: Path,
    elf_path: Path,
    *,
    verbose: bool,
    success_marker: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(qemu),
        "-M",
        "mps3-an547",
        "-cpu",
        "cortex-m55",
        "-kernel",
        str(elf_path),
        "-semihosting",
        "-semihosting-config",
        "enable=on,target=auto",
        "-serial",
        "mon:stdio",
        "-nographic",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_chunks: list[str] = []
    marker_detected = False
    failure_detected = False
    fatal_detected = False
    deadline = time.monotonic() + timeout

    assert process.stdout is not None  # for type checkers

    while True:
        if process.poll() is not None:
            # Process exited on its own; capture remaining output.
            remainder = process.stdout.read()
            if remainder:
                output_chunks.append(remainder)
                if verbose:
                    sys.stdout.write(remainder)
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        ready, _, _ = select.select([process.stdout], [], [], max(remaining, 0.0))
        if not ready:
            continue

        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if not line:
            continue

        output_chunks.append(line)
        if verbose:
            sys.stdout.write(line)

        if success_marker and success_marker in line:
            marker_detected = True
            break
        if "FAIL" in line:
            failure_detected = True
            break
        if "fatal: Lockup" in line:
            fatal_detected = True
            break

    if marker_detected:
        # Stop QEMU once success marker is seen to avoid waiting for watchdog timeouts.
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        return subprocess.CompletedProcess(cmd, 0, "".join(output_chunks), "")

    # Either timed out or saw an explicit failure; ensure QEMU terminates.
    process.terminate()
    try:
        stdout_tail, _ = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_tail, _ = process.communicate()
    if stdout_tail:
        output_chunks.append(stdout_tail)

    stdout_text = "".join(output_chunks)
    exit_code = process.returncode if process.returncode is not None else -1

    if failure_detected or fatal_detected:
        return subprocess.CompletedProcess(cmd, exit_code or 1, stdout_text, "")

    # Timed out or exited without explicit marker; treat as success but annotate return code.
    if exit_code not in (0, None):
        return subprocess.CompletedProcess(cmd, exit_code, stdout_text, "")
    return subprocess.CompletedProcess(cmd, 0, stdout_text, "")


def run_qemu(
    qemu: Path,
    elf_path: Path,
    *,
    verbose: bool,
    success_marker: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Backward-compatible helper that mirrors the old functional API."""

    manager = QemuManager(Path(qemu), timeout=timeout, verbose=verbose)
    return manager.run(Path(elf_path), success_marker=success_marker)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        dest="operations",
        action="append",
        choices=AVAILABLE_OPERATION_CHOICES,
        help="STM32N6 operation harness to run (default: conv)",
    )
    parser.add_argument("--zig", help="Path to a Zig binary for cross-compilation")
    parser.add_argument("--clang", help="Path to a Clang binary for cross-compilation")
    parser.add_argument(
        "--arm-prefix",
        help="GNU Arm Embedded Toolchain prefix (default: arm-none-eabi)",
    )
    parser.add_argument("--host-gcc", help="Path to a host GCC fallback compiler")
    parser.add_argument("--qemu", help="Path to qemu-system-arm")
    parser.add_argument("--cmsis-include", help="Directory containing arm_math.h")
    parser.add_argument(
        "--cmsis-nn-include",
        help="Directory containing arm_nnfunctions.h (defaults to sibling of --cmsis-include)",
    )
    parser.add_argument(
        "--cmsis-convolve",
        "--cmsis-source",
        dest="cmsis_convolve",
        help="Override the path to arm_convolve_s8.c (defaults to CMSIS or stub)",
    )
    parser.add_argument("--keep-build", action="store_true", help="Keep the build directory")
    parser.add_argument("--verbose", action="store_true", help="Stream QEMU output as it runs")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run each firmware variant when measuring timing",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=3.0,
        help="Allow each QEMU instance to run for this many seconds before the harness terminates it",
    )
    parser.add_argument(
        "--beer",
        action="store_true",
        help="Build and run the Beer model firmware variants",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.repeat <= 0:
        raise ToolchainError("--repeat must be at least 1")

    zig_path = detect_zig(args.zig)
    arm_prefix = detect_arm_gcc(args.arm_prefix)
    clang_path = detect_clang(args.clang)
    host_gcc = detect_host_gcc(args.host_gcc)

    selection: ToolchainSelection | None = None
    if arm_prefix is not None:
        selection = ToolchainSelection(ArmGccToolchain(arm_prefix), BuildFlavor.BARE_METAL)
    elif zig_path is not None:
        selection = ToolchainSelection(ZigToolchain(zig_path), BuildFlavor.BARE_METAL)
    elif clang_path is not None:
        selection = ToolchainSelection(ClangToolchain(clang_path), BuildFlavor.BARE_METAL)
    elif host_gcc is not None:
        selection = ToolchainSelection(HostGccToolchain(host_gcc), BuildFlavor.HOST)
    else:
        raise ToolchainError(
            "no cross compiler available; install Zig 0.14, the GNU Arm Embedded toolchain, or provide --host-gcc"
        )

    assert selection is not None

    if selection.flavor == BuildFlavor.HOST:
        qemu_path = detect_qemu(args.qemu) if args.qemu else None
    else:
        qemu_path = detect_qemu(args.qemu)
        if qemu_path is None:
            raise ToolchainError(
                "qemu-system-arm not found; install QEMU or set the --qemu flag"
            )

    include_dir = find_arm_math_header(args.cmsis_include)
    nn_include = find_arm_nn_header(args.cmsis_nn_include, include_dir)
    convolve_source = find_arm_convolve_source(args.cmsis_convolve, nn_include)
    cmsis_bundle = CmsisBundle(include_dir, nn_include, convolve_source)
    requested_operations = args.operations if args.operations else ["conv"]
    operations = instantiate_operations(requested_operations)
    if not operations:
        raise ToolchainError("no STM32N6 operations selected")

    print(f"Using toolchain: {selection.toolchain.describe()}")
    if qemu_path is not None:
        print(f"Using QEMU: {qemu_path}")
    else:
        print("Using host execution (no QEMU)")
    print(f"CMSIS DSP include path: {include_dir}")
    print(f"CMSIS NN include path: {nn_include}")
    print(f"arm_convolve_s8 source: {convolve_source}")

    build_ctx = (
        tempfile.TemporaryDirectory(prefix="stm32n6-qemu-")
        if not args.keep_build
        else None
    )
    qemu_manager = (
        QemuManager(qemu_path, timeout=args.run_seconds, verbose=args.verbose)
        if qemu_path is not None
        else None
    )

    operation_results: list[tuple[FirmwareOperation, list[CaseTiming]]] = []
    beer_timings: list[CaseTiming] | None = None
    beer_operation: BeerOperation | None = None
    try:
        build_dir = Path(build_ctx.name) if build_ctx is not None else (REPO_ROOT / "build" / "stm32n6_qemu")
        if build_ctx is None:
            build_dir.mkdir(parents=True, exist_ok=True)

        for operation in operations:
            print(f"\n[op] {operation.display_name}")
            cases = operation.build_cases(cmsis_bundle)
            runner = HarnessRunner(
                operation=operation,
                selection=selection,
                qemu=qemu_manager,
                build_dir=build_dir,
                repeat=args.repeat,
                verbose=args.verbose,
            )
            operation_results.append((operation, runner.run_cases(cases)))

        if args.beer:
            if selection.flavor == BuildFlavor.HOST:
                raise ToolchainError(
                    "beer firmware requires a bare-metal toolchain; rerun with --arm-prefix, --zig, or --clang"
                )
            if qemu_manager is None:
                raise ToolchainError(
                    "beer firmware requires qemu-system-arm; provide --qemu or install QEMU"
                )
            beer_lib = ensure_beer_library(zig_path=zig_path)
            beer_operation = BeerOperation(beer_lib)
            beer_cases = beer_operation.build_cases(cmsis_bundle)
            beer_runner = HarnessRunner(
                operation=beer_operation,
                selection=selection,
                qemu=qemu_manager,
                build_dir=build_dir,
                repeat=args.repeat,
                verbose=args.verbose,
            )
            try:
                beer_timings = beer_runner.run_cases(beer_cases)
            except ToolchainError as exc:
                raise ToolchainError(
                    "beer firmware requires a bare-metal toolchain; rerun with --arm-prefix, --zig, or --clang"
                ) from exc
    finally:
        if build_ctx is not None:
            build_ctx.cleanup()

    for operation, case_timings in operation_results:
        print(f"\nAll {operation.display_name} cases passed.")
        operation.print_summary(case_timings)
    if beer_timings:
        assert beer_operation is not None
        print(f"\nAll {beer_operation.display_name} cases passed.")
        beer_operation.print_summary(beer_timings)
    return 0


def ensure_beer_library(*, zig_path: Path | None) -> Path:
    if BEER_LIB_PATH.exists():
        return BEER_LIB_PATH

    if zig_path is None:
        raise ToolchainError(
            "beer firmware requires zig-out/beer/libzant.a; rerun with --zig or build it manually"
        )

    env = os.environ.copy()
    env.setdefault("ZANT_FBA_SIZE_KB", "320")
    env.setdefault("ZANT_FBA_SECTION", ".tensor_pool")

    cmd = [
        str(zig_path),
        "build",
        "lib",
        "-Dmodel=beer",
        "-Ddynamic=true",
        "-Ddo_export=true",
        "-Dfuse=true",
        "-Dtarget=thumb-freestanding",
        "-Dcpu=cortex_m55",
        "-Doptimize=ReleaseSmall",
        "-Dstm32n6_accel=true",
        "-Dstm32n6_use_cmsis=true",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    if BEER_LIB_PATH.exists():
        return BEER_LIB_PATH

    raise ToolchainError(
        f"zig build completed but {BEER_LIB_PATH} was not produced"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
