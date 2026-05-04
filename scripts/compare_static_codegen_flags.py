#!/usr/bin/env python3
"""Run static codegen with every explicit planner flag and open an HTML report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import shutil
import shlex
import subprocess
import sys
import time
import webbrowser
from html import escape
from typing import Any


EXPLICIT_STATIC_PLANNING_OPTIONS = [
    "pressure_then_size",
    "pressure_then_liveness",
    "liveness_first",
    "size_first",
    "first_step",
]

STATIC_PLANNING_OPTIONS = [
    *EXPLICIT_STATIC_PLANNING_OPTIONS,
    *[
        f"{option}_inverse_first_step"
        for option in EXPLICIT_STATIC_PLANNING_OPTIONS
    ],
]

TYPE_BYTES = {
    "bool": 1,
    "u8": 1,
    "i8": 1,
    "u16": 2,
    "i16": 2,
    "f16": 2,
    "u32": 4,
    "i32": 4,
    "f32": 4,
    "u64": 8,
    "i64": 8,
    "f64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run `zig build lib-gen` for a model with every explicit "
            "-Dstatic_planning ordering and its _inverse_first_step variant, "
            "then open a local HTML comparison page."
        ),
        epilog="Extra Zig build arguments can be placed after `--`, for example: %(prog)s --model=beer -- -Dfuse=true",
    )
    parser.add_argument("--model", required=True, help="Model name, for example `beer`.")
    parser.add_argument(
        "--model-path",
        help="Optional ONNX model path. Defaults to datasets/models/<model>/<model>.onnx.",
    )
    parser.add_argument("--zig", default="zig", help="Zig executable to run.")
    parser.add_argument(
        "--build-arg",
        action="append",
        default=[],
        help="Extra argument passed to `zig build`; repeat for multiple values.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Per-planner timeout in seconds.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed codegen run.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the report without opening it in the browser.",
    )
    args, extra_build_args = parser.parse_known_args()
    args.extra_build_args = extra_build_args
    return args


def clean_extra_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def sh_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def report_command(command: list[str]) -> list[str]:
    return [
        part
        for part in command
        if part != "-Dgenerated_path=generated"
    ]


def summarize_failure(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in f"{stderr}\n{stdout}".splitlines() if line.strip()]
    for line in lines:
        if "panic:" in line or line.startswith("error:") or " error:" in line:
            return line
    return lines[-1] if lines else "zig build failed"


def element_type_bytes(element_type: Any) -> int:
    if element_type is None:
        return 1
    return TYPE_BYTES.get(str(element_type), 1)


def load_plan(plan_path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with plan_path.open("r", encoding="utf-8") as plan_file:
        plan = json.load(plan_file)

    if isinstance(plan, list):
        return {}, plan

    if isinstance(plan, dict):
        tensors = plan.get("tensors")
        if not isinstance(tensors, list):
            raise ValueError("static memory plan JSON does not contain a tensors list")
        metadata = plan.get("metadata")
        return metadata if isinstance(metadata, dict) else {}, tensors

    raise ValueError("static memory plan JSON must be a list or an object")


def compute_plan_metrics(plan_path: pathlib.Path) -> dict[str, Any]:
    metadata, tensors = load_plan(plan_path)

    buffers: dict[tuple[Any, Any], dict[str, Any]] = {}
    peak_events: dict[int, int] = {}
    max_end_step = 0
    unknown_types: set[str] = set()

    for tensor in tensors:
        backing_buffer = tensor.get("backing_buffer") or {}
        element_type = backing_buffer.get("element_type")
        byte_width = element_type_bytes(element_type)
        if element_type is not None and str(element_type) not in TYPE_BYTES:
            unknown_types.add(str(element_type))

        buffer_id = backing_buffer.get("id")
        if buffer_id is not None:
            buffers[(buffer_id, element_type)] = backing_buffer

        start = int(backing_buffer.get("start_borrow", 0))
        end = int(backing_buffer.get("end_borrow", start))
        max_end_step = max(max_end_step, end)

        tensor_bytes = int(tensor.get("size", 0)) * byte_width
        peak_events[start] = peak_events.get(start, 0) + tensor_bytes
        peak_events[end + 1] = peak_events.get(end + 1, 0) - tensor_bytes

    total_buffer_bytes = sum(
        int(buffer.get("size", 0)) * element_type_bytes(buffer.get("element_type"))
        for buffer in buffers.values()
    )

    current_live_bytes = 0
    peak_live_bytes = 0
    peak_live_step = 0
    for step in sorted(peak_events):
        current_live_bytes += peak_events[step]
        if current_live_bytes > peak_live_bytes:
            peak_live_bytes = current_live_bytes
            peak_live_step = step

    overhead_percent = (
        ((total_buffer_bytes / peak_live_bytes) - 1.0) * 100.0
        if peak_live_bytes
        else math.nan
    )

    return {
        "planner": metadata.get("planner", "unknown"),
        "node_count": metadata.get("node_count"),
        "tensor_count": len(tensors),
        "buffer_count": len(buffers),
        "total_buffer_bytes": total_buffer_bytes,
        "peak_live_bytes": peak_live_bytes,
        "peak_live_step": peak_live_step,
        "max_end_step": max_end_step,
        "overhead_percent": overhead_percent,
        "unknown_types": sorted(unknown_types),
    }


def run_codegen(
    *,
    zig: str,
    model: str,
    model_path: str | None,
    option: str,
    model_output_dir: pathlib.Path,
    extra_args: list[str],
    timeout: float | None,
) -> dict[str, Any]:
    comparison_dir = model_output_dir / "static_planning_comparison"
    plans_dir = comparison_dir / "plans"
    logs_dir = comparison_dir / "logs"
    generated_arg = pathlib.Path("generated")
    generated_model_dir = model_output_dir
    generated_plan_path = generated_model_dir / f"memory_allocation_{model}.json"
    plan_path = plans_dir / f"memory_allocation_{model}_{option}.json"
    log_path = logs_dir / f"{option}.log"

    plans_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if generated_plan_path.exists():
        generated_plan_path.unlink()

    command = [
        zig,
        "build",
        "lib-gen",
        f"-Dmodel={model}",
        "-Ddynamic=false",
        f"-Dstatic_planning={option}",
        f"-Dgenerated_path={generated_arg.as_posix()}",
    ]
    if model_path:
        command.append(f"-Dmodel_path={model_path}")
    command.extend(extra_args)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.monotonic() - started
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        log_path.write_text(
            f"$ {sh_join(command)}\n\n"
            "===== stdout =====\n"
            f"{stdout}\n\n"
            "===== stderr =====\n"
            f"{stderr}\n",
            encoding="utf-8",
        )

        result = {
            "option": option,
            "status": "ok" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "duration_seconds": duration,
            "command": command,
            "command_text": sh_join(command),
            "report_command_text": sh_join(report_command(command)),
            "generated_dir": str(generated_model_dir),
            "plan_path": str(plan_path),
            "log_path": str(log_path),
            "error": "",
        }

        if completed.returncode != 0:
            result["error"] = (stderr or stdout).strip().splitlines()[-1:] or [
                "zig build failed"
            ]
            result["error"] = summarize_failure(stdout, stderr)
            return result

        if not generated_plan_path.exists():
            result["status"] = "failed"
            result["error"] = f"expected plan was not created: {generated_plan_path}"
            return result

        shutil.copyfile(generated_plan_path, plan_path)
        result.update(compute_plan_metrics(plan_path))
        return result
    except subprocess.TimeoutExpired as err:
        duration = time.monotonic() - started
        log_path.write_text(
            f"$ {sh_join(command)}\n\nTimed out after {timeout} seconds.\n"
            f"{err.stdout or ''}\n{err.stderr or ''}\n",
            encoding="utf-8",
        )
        return {
            "option": option,
            "status": "timeout",
            "return_code": None,
            "duration_seconds": duration,
            "command": command,
            "command_text": sh_join(command),
            "report_command_text": sh_join(report_command(command)),
            "generated_dir": str(generated_model_dir),
            "plan_path": str(plan_path),
            "log_path": str(log_path),
            "error": f"timed out after {timeout} seconds",
        }


def format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{int(value)} B"


def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}%"


def html_report(model: str, results: list[dict[str, Any]], report_path: pathlib.Path) -> str:
    successful = [result for result in results if result.get("status") == "ok"]
    sorted_successful = sorted(
        successful,
        key=lambda r: (r["total_buffer_bytes"], r["duration_seconds"]),
    )
    best = sorted_successful[0] if sorted_successful else None
    fastest = (
        min(successful, key=lambda r: r["duration_seconds"]) if successful else None
    )
    max_total = max(
        [result.get("total_buffer_bytes", 0) for result in successful] or [1]
    )

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for index, result in enumerate(results, 1):
        is_best = best and result["option"] == best["option"]
        status = result.get("status", "unknown")
        status_class = "ok" if status == "ok" else "bad"
        plan_link = (
            pathlib.Path(result["plan_path"]).resolve().as_uri()
            if result.get("status") == "ok"
            else ""
        )
        log_link = pathlib.Path(result["log_path"]).resolve().as_uri()
        bar_width = (
            max(4, int(result.get("total_buffer_bytes", 0) / max_total * 100))
            if status == "ok"
            else 0
        )
        rows.append(
            f"""
            <tr class="{'winner' if is_best else ''}" data-option="{escape(result['option'])}">
              <td><input type="radio" name="chosen" value="{escape(result['option'])}" {'checked' if is_best else ''}></td>
              <td>{index}</td>
              <td><code>{escape(result['option'])}</code></td>
              <td><span class="status {status_class}">{escape(status)}</span></td>
              <td>{escape(format_bytes(result.get('total_buffer_bytes')))}</td>
              <td>{escape(format_bytes(result.get('peak_live_bytes')))}</td>
              <td>{escape(format_percent(result.get('overhead_percent')))}</td>
              <td>{escape(str(result.get('buffer_count', 'n/a')))}</td>
              <td>{escape(str(result.get('tensor_count', 'n/a')))}</td>
              <td>{escape(str(result.get('node_count', 'n/a')))}</td>
              <td>{escape(str(result.get('planner', 'n/a')))}</td>
              <td>{result.get('duration_seconds', 0.0):.2f}s</td>
              <td><div class="bar"><span style="width: {bar_width}%"></span></div></td>
              <td>{f'<a href="{plan_link}">plan</a>' if plan_link else ''} <a href="{log_link}">log</a></td>
            </tr>
            """
        )

    payload = json.dumps(results).replace("</", "<\\/")
    best_option = best["option"] if best else ""
    best_command = best["report_command_text"] if best else ""
    fastest_text = (
        f"{fastest['option']} in {fastest['duration_seconds']:.2f}s"
        if fastest
        else "n/a"
    )
    best_text = (
        f"{best['option']} ({format_bytes(best['total_buffer_bytes'])})"
        if best
        else "n/a"
    )
    report_uri = report_path.resolve().as_uri()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Static Codegen Planner Comparison - {escape(model)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #182027;
      --muted: #60707f;
      --line: #d8e0e7;
      --soft: #f5f7f9;
      --brand: #0f766e;
      --brand-soft: #d8f3ee;
      --accent: #b45309;
      --bad: #b42318;
      --bad-soft: #fee4e2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fbfa 0%, #ffffff 100%);
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    .meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{ padding: 24px 36px 40px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .card p {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .selected {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      margin-bottom: 16px;
    }}
    .selected code {{
      display: block;
      overflow-x: auto;
      white-space: nowrap;
      font-size: 13px;
    }}
    button {{
      border: 1px solid #8fbdb7;
      border-radius: 6px;
      background: var(--brand);
      color: white;
      font-weight: 700;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1120px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #eef4f3;
      color: #33434f;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:hover {{ background: #f7faf9; }}
    tr.winner {{ background: var(--brand-soft); }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    a {{ color: #0b5cad; text-decoration: none; font-weight: 700; }}
    a:hover {{ text-decoration: underline; }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
    }}
    .status.ok {{ background: var(--brand-soft); color: #075e56; }}
    .status.bad {{ background: var(--bad-soft); color: var(--bad); }}
    .bar {{
      width: 160px;
      height: 9px;
      background: #e6edf2;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--brand), var(--accent));
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Static Codegen Planner Comparison: {escape(model)}</h1>
    <div class="meta">Generated {escape(now)} from <code>{escape(str(report_path))}</code></div>
  </header>
  <main>
    <section class="cards" aria-label="Summary">
      <div class="card"><h2>Lowest Static Allocation</h2><p>{escape(best_text)}</p></div>
      <div class="card"><h2>Fastest Codegen</h2><p>{escape(fastest_text)}</p></div>
      <div class="card"><h2>Successful Runs</h2><p>{len(successful)} / {len(results)}</p></div>
    </section>
    <section class="selected" aria-label="Selected planner command">
      <div>
        <strong>Selected flag: <span id="selected-option">{escape(best_option)}</span></strong>
        <code id="selected-command">{escape(best_command)}</code>
      </div>
      <button type="button" id="copy-command">Copy command</button>
    </section>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Choose</th>
            <th>#</th>
            <th>Flag</th>
            <th>Status</th>
            <th>Static Allocation</th>
            <th>Peak Live</th>
            <th>Overhead</th>
            <th>Buffers</th>
            <th>Tensors</th>
            <th>Nodes</th>
            <th>Planner</th>
            <th>Build Time</th>
            <th>Scale</th>
            <th>Files</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    <p class="note">
      Static allocation is the total size reserved by unique backing buffers.
      Peak live is the largest simultaneous tensor payload estimate from the emitted plan.
      The best row is selected by lowest static allocation, then lowest codegen time.
    </p>
  </main>
  <script>
    const results = {payload};
    const byOption = new Map(results.map((result) => [result.option, result]));
    const optionLabel = document.getElementById("selected-option");
    const commandLabel = document.getElementById("selected-command");
    function choose(option) {{
      const result = byOption.get(option);
      if (!result) return;
      optionLabel.textContent = option;
      commandLabel.textContent = result.report_command_text || result.command_text || "";
      document.querySelectorAll("tbody tr").forEach((row) => {{
        row.classList.toggle("winner", row.dataset.option === option);
      }});
    }}
    document.querySelectorAll('input[name="chosen"]').forEach((input) => {{
      input.addEventListener("change", () => choose(input.value));
    }});
    document.querySelectorAll("tbody tr").forEach((row) => {{
      row.addEventListener("click", (event) => {{
        if (event.target.tagName === "A") return;
        const input = row.querySelector('input[name="chosen"]');
        input.checked = true;
        choose(input.value);
      }});
    }});
    document.getElementById("copy-command").addEventListener("click", async () => {{
      await navigator.clipboard.writeText(commandLabel.textContent);
    }});
  </script>
  <!-- Report URI: {escape(report_uri)} -->
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    extra_args = [*args.build_arg, *clean_extra_args(args.extra_build_args)]
    model_output_dir = pathlib.Path("generated") / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running static codegen comparison for model `{args.model}`")
    print("Planner flags:")
    for option in STATIC_PLANNING_OPTIONS:
        print(f"  - {option}")

    results: list[dict[str, Any]] = []
    for index, option in enumerate(STATIC_PLANNING_OPTIONS, 1):
        print(f"[{index}/{len(STATIC_PLANNING_OPTIONS)}] {option} ... ", end="", flush=True)
        result = run_codegen(
            zig=args.zig,
            model=args.model,
            model_path=args.model_path,
            option=option,
            model_output_dir=model_output_dir,
            extra_args=extra_args,
            timeout=args.timeout,
        )
        results.append(result)
        if result["status"] == "ok":
            print(
                f"ok, {format_bytes(result['total_buffer_bytes'])}, "
                f"{result['duration_seconds']:.2f}s"
            )
        else:
            print(f"{result['status']}: {result.get('error', 'unknown error')}")
            if args.fail_fast:
                break

    report_path = model_output_dir / "static_codegen_planner_comparison.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_report(args.model, results, report_path), encoding="utf-8")

    print(f"\nReport written to {report_path}")
    if not args.no_open:
        opened = webbrowser.open(report_path.resolve().as_uri())
        if not opened:
            print("Could not open a browser automatically.")
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
