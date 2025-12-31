#!/usr/bin/env python3
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "build/beer_comparison/comparison_results.txt"
README = REPO / "README.md"

def main() -> int:
    if not RESULTS.exists():
        print(f"no results file: {RESULTS}")
        return 0
    content = RESULTS.read_text()
    ref_match = re.search(r"beer_reference:\s*([\d.]+) ms", content)
    cms_match = re.search(r"beer_cmsis_nn:\s*([\d.]+) ms", content)
    if not (ref_match and cms_match):
        print("results not found in comparison_results.txt")
        return 0
    ref = float(ref_match.group(1))
    cms = float(cms_match.group(1))
    improvement = ref - cms
    pct = (improvement / ref) * 100.0 if ref > 0 else 0.0

    # Update or insert timings block at the top, right after the badges div if present
    readme_text = README.read_text() if README.exists() else ""

    block_start = "<!-- BEER_TIMINGS_START -->"
    block_end = "<!-- BEER_TIMINGS_END -->"
    new_block = (
        f"{block_start}\n"
        f"Beer model timing (QEMU, Cortex-M55):\n\n"
        f"- Reference: {ref:.2f} ms\n"
        f"- CMSIS-NN: {cms:.2f} ms\n"
        f"- Improvement: {improvement:.2f} ms ({pct:.1f}%)\n"
        f"{block_end}\n\n"
    )

    # Find badges closing tag </div> to insert after; otherwise after first heading line
    insert_pos = readme_text.find(block_start)
    if insert_pos != -1:
        insert_pos += len("</div>\n") if readme_text.startswith("#") else len("</div>")
        # Ensure we insert after the line with </div>
        # Move to end of that line
        nl = readme_text.find("\n", insert_pos)
        if nl != -1:
            insert_pos = nl + 1
    else:
        # Insert after the first line (typically the H1)
        nl = readme_text.find("\n")
        insert_pos = nl + 1 if nl != -1 else 0

        # Remove any existing block anywhere
    readme_text = re.sub(
        rf"{re.escape(block_start)}[\s\S]*?{re.escape(block_end)}\n?",
        "",
        readme_text,
        flags=re.M,
    )

    updated = readme_text[:insert_pos] + new_block + readme_text[insert_pos:]
    README.write_text(updated)
    print("README.md updated with beer timings")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


