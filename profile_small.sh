#!/usr/bin/env bash
set -euo pipefail

# Models to profile
models=("tiny_dscnn_XS" "tiny_dscnn_S" "tiny_dscnn_M" "tiny_dscnn_L" "tiny_dscnn_XL")   # add more:  "mnist-8" "mobilenet_v2" "beer" "fomo8" "coco80_q"

# Zig optimization modes
optimize_modes=( "ReleaseSmall" )

# Zig static flags (adjust if needed)
STATIC_FLAGS="-Ddynamic=false -Dstatic_planning=true -Duse_tensor_pool=true"

mkdir -p profiling

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Build helper: returns 0/1 if binary exists
exe_size() {
  if [[ -f "./zig-out/bin/main_profiling_target" ]]; then
    ls -lh ./zig-out/bin/main_profiling_target | awk '{print $5}'
  else
    echo "N/A"
  fi
}

echo "== profiling start: $(timestamp) =="
echo "Models: ${models[*]}"
echo

for model in "${models[@]}"; do

  # ************************************************************************************************************************ 
  # ************************************************************************************************************************
  # ******************************************************** STATIC ********************************************************
  # ************************************************************************************************************************
  # ************************************************************************************************************************ 

  outdir="profiling/${model}/static"
  mkdir -p "$outdir"

  report="${outdir}/${model}_static_profiling_report.txt"
  : > "$report"  # truncate

  {
    echo "====================================="
    echo "STATIC PROFILING REPORT"
    echo "Model: ${model}"
    echo "Date:  $(timestamp)"
    echo "====================================="
    echo
  } >> "$report"

  echo "Codegen STATIC..."
  zig build lib-gen -Dmodel="${model}" -Ddo_export -Dfuse ${STATIC_FLAGS}

  # Clean previous Valgrind outputs for clarity
  rm -f callgrind.out.* massif.out.* valgrind_memcheck.log || true

  echo "build-main STATIC..."
  # Build library + main target (static)
  zig build build-main -Dmodel="${model}" -Doptimize=ReleaseSmall -Dfuse

  # Verify exe
  if [[ ! -f "./zig-out/bin/main_profiling_target" ]]; then
    {
      echo
      echo "-------------------------------------"
      echo "ERROR: Executable not found "
      echo "-------------------------------------"
      echo
    } >> "$report"
    continue
  fi

  binsize="$(exe_size)"

  {
    echo "====================================="
    echo "PROFILING: ${model} "
    echo "Date: $(timestamp)"
    echo "Executable size: ${binsize}"
    echo "====================================="
    echo
  } >> "$report"

  # ---------- NEW: capture ONE plain run's program output ----------
  {
    echo "=== PROGRAM OUTPUT () ==="
    # Indent each line for readability; remove 'sed' if you prefer raw.
    ./zig-out/bin/main_profiling_target 2>&1 | sed 's/^/    /'
    echo
  } >> "$report"
  # ----------------------------------------------------------------

  # Deterministic filenames (use PID for uniqueness too)
  pid=$$
  cg_file="${outdir}/callgrind.out.${model}.${pid}"
  ms_file="${outdir}/massif.out.${model}.${pid}"
  memlog="${outdir}/${model}_valgrind_memcheck.log"

  # 1) Valgrind Memcheck (append only summaries to report to keep it short)
  echo "  - Memcheck..."
  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
            --log-file="${memlog}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== MEMORY LEAK ANALYSIS  ==="
    grep -A 10 "LEAK SUMMARY:" "${memlog}" || echo "No leak summary found"
    echo
    echo "=== HEAP SUMMARY  ==="
    grep -A 5 "HEAP SUMMARY:" "${memlog}" || echo "No heap summary found"
    echo
    echo "=== ERROR SUMMARY  ==="
    grep "ERROR SUMMARY:" "${memlog}" || echo "No error summary found"
    echo
  } >> "$report"

  # 2) Callgrind (top functions)
  echo "  - Callgrind..."
  valgrind --tool=callgrind --callgrind-out-file="${cg_file}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== CALLGRIND ANALYSIS ==="
    echo "Callgrind file: $(basename "${cg_file}")"
    if command -v callgrind_annotate >/dev/null 2>&1; then
      callgrind_annotate --threshold=1 "${cg_file}" 2>/dev/null | head -n 30 || echo "Failed to annotate"
    else
      echo "callgrind_annotate not found; skipping annotation"
    fi
    echo
  } >> "$report"

  # 3) Massif (heap timeline)
  echo "  - Massif..."
  valgrind --tool=massif --time-unit=ms --massif-out-file="${ms_file}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== MASSIF ANALYSIS  ==="
    echo "Massif file: $(basename "${ms_file}")"
    if command -v ms_print >/dev/null 2>&1; then
      ms_print "${ms_file}" 2>/dev/null | head -n 50 || echo "Failed to print massif"
    else
      echo "ms_print not found; skipping ms_print output"
    fi
    echo
    echo "-------------------------------------"
    echo
  } >> "$report"

  # Tiny, readable summary section
  {
    echo
    echo "====================================="
    echo "SUMMARY: ${model} (static)"
    echo "====================================="
    printf "%-15s | %-15s\n" "Optimization" "Binary Size"
    printf "%-15s-+-%-15s\n" "---------------" "---------------"
    
    size="$(exe_size)"
    printf "%-15s | %-15s\n" "${size}"
    
    echo
    echo "Artifacts:"
    echo "  - ${report}"
    echo "  - ${outdir}/*_valgrind_memcheck.log"
    echo "  - ${outdir}/callgrind.out.*"
    echo "  - ${outdir}/massif.out.*"
    echo
  } >> "$report"

  echo "Report written: ${report}"

  # ************************************************************************************************************************* 
  # ************************************************************************************************************************* 
  # ******************************************************** DYNAMIC ********************************************************
  # *************************************************************************************************************************
  # ************************************************************************************************************************* 

  outdir="profiling/${model}/dynamic"
  mkdir -p "$outdir"

  report="${outdir}/${model}_dynamic_profiling_report.txt"
  : > "$report"  # truncate

  {
    echo "====================================="
    echo "DYNAMIC PROFILING REPORT"
    echo "Model: ${model}"
    echo "Date:  $(timestamp)"
    echo "====================================="
    echo
  } >> "$report"

  echo "Codegen DYNAMIC..."
  zig build lib-gen -Dmodel="${model}" -Ddo_export -Dfuse

  # Clean previous Valgrind outputs for clarity
  rm -f callgrind.out.* massif.out.* valgrind_memcheck.log || true

  echo "build-main DYNAMIC..."
  # Build library + main target (dynamic)
  zig build build-main -Dmodel="${model}" -Doptimize=ReleaseSmall

  # Verify exe
  if [[ ! -f "./zig-out/bin/main_profiling_target" ]]; then
    {
      echo
      echo "-------------------------------------"
      echo "ERROR: Executable not found "
      echo "-------------------------------------"
      echo
    } >> "$report"
    continue
  fi

  binsize="$(exe_size)"

  {
    echo "====================================="
    echo "PROFILING: ${model} "
    echo "Date: $(timestamp)"
    echo "Executable size: ${binsize}"
    echo "====================================="
    echo
  } >> "$report"

  # ---------- NEW: capture ONE plain run's program output ----------
  {
    echo "=== PROGRAM OUTPUT () ==="
    # Indent each line for readability; remove 'sed' if you prefer raw.
    ./zig-out/bin/main_profiling_target 2>&1 | sed 's/^/    /'
    echo
  } >> "$report"
  # ----------------------------------------------------------------

  # Deterministic filenames (use PID for uniqueness too)
  pid=$$
  cg_file="${outdir}/callgrind.out.${model}.${pid}"
  ms_file="${outdir}/massif.out.${model}.${pid}"
  memlog="${outdir}/${model}_valgrind_memcheck.log"

  # 1) Valgrind Memcheck (append only summaries to report to keep it short)
  echo "  - Memcheck..."
  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
            --log-file="${memlog}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== MEMORY LEAK ANALYSIS  ==="
    grep -A 10 "LEAK SUMMARY:" "${memlog}" || echo "No leak summary found"
    echo
    echo "=== HEAP SUMMARY  ==="
    grep -A 5 "HEAP SUMMARY:" "${memlog}" || echo "No heap summary found"
    echo
    echo "=== ERROR SUMMARY  ==="
    grep "ERROR SUMMARY:" "${memlog}" || echo "No error summary found"
    echo
  } >> "$report"

  # 2) Callgrind (top functions)
  echo "  - Callgrind..."
  valgrind --tool=callgrind --callgrind-out-file="${cg_file}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== CALLGRIND ANALYSIS ==="
    echo "Callgrind file: $(basename "${cg_file}")"
    if command -v callgrind_annotate >/dev/null 2>&1; then
      callgrind_annotate --threshold=1 "${cg_file}" 2>/dev/null | head -n 30 || echo "Failed to annotate"
    else
      echo "callgrind_annotate not found; skipping annotation"
    fi
    echo
  } >> "$report"

  # 3) Massif (heap timeline)
  echo "  - Massif..."
  valgrind --tool=massif --time-unit=ms --massif-out-file="${ms_file}" \
            ./zig-out/bin/main_profiling_target >/dev/null 2>&1

  {
    echo "=== MASSIF ANALYSIS  ==="
    echo "Massif file: $(basename "${ms_file}")"
    if command -v ms_print >/dev/null 2>&1; then
      ms_print "${ms_file}" 2>/dev/null | head -n 50 || echo "Failed to print massif"
    else
      echo "ms_print not found; skipping ms_print output"
    fi
    echo
    echo "-------------------------------------"
    echo
  } >> "$report"

  # Tiny, readable summary section
  {
    echo
    echo "====================================="
    echo "SUMMARY: ${model} (dynamic)"
    echo "====================================="
    printf "%-15s | %-15s\n" "Optimization" "Binary Size"
    printf "%-15s-+-%-15s\n" "---------------" "---------------"
    
    size="$(exe_size)"
    printf "%-15s | %-15s\n" "${size}"
    
    echo
    echo "Artifacts:"
    echo "  - ${report}"
    echo "  - ${outdir}/*_valgrind_memcheck.log"
    echo "  - ${outdir}/callgrind.out.*"
    echo "  - ${outdir}/massif.out.*"
    echo
  } >> "$report"

  echo "Report written: ${report}"

done

echo "== Done =="
