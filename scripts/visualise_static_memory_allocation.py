#!/usr/bin/env python3
# Before running: pip install pandas plotly
# Run with the name of a json file containing a description of a static memory plan as a command line argument
import html
import json
import sys

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BUFFER_PREFIX = "backing_buffer"
START_T = f"{BUFFER_PREFIX}.start_borrow"
END_T = f"{BUFFER_PREFIX}.end_borrow"
TENSOR_SIZE = "size"
BUFFER_ID = f"{BUFFER_PREFIX}.id"
BUFFER_SIZE = f"{BUFFER_PREFIX}.size"
BUFFER_TYPE = f"{BUFFER_PREFIX}.element_type"
BORROW_DURATION = "duration"
TENSOR_NAME = "name"
LABEL = "label"
USAGE = "usage"

if len(sys.argv) != 2:
    print(
        "Usage: ./visualise_static_memory_allocation.py"
        " <static_memory_plan.json>"
    )
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as plan_file:
    static_memory_plan = json.load(plan_file)

if isinstance(static_memory_plan, list):
    metadata = {}
    tensor_rows = static_memory_plan
elif isinstance(static_memory_plan, dict):
    metadata = static_memory_plan.get("metadata", {})
    tensor_rows = static_memory_plan.get("tensors")
    if not isinstance(tensor_rows, list):
        raise ValueError("Expected static memory plan JSON to contain a 'tensors' list")
else:
    raise ValueError("Expected static memory plan JSON to be either a list or an object")

planner = metadata.get("planner", "unknown")
flags = metadata.get(
    "flags",
    metadata.get("static_planning_flags", metadata.get("static_planning_option", "unknown")),
)
node_count = metadata.get("node_count")

df = pd.DataFrame(tensor_rows)
df = df.join(
    pd.json_normalize(df[BUFFER_PREFIX]).add_prefix(f"{BUFFER_PREFIX}.")
)

df.sort_values(START_T, inplace=True)

# 0.2 and 0.1 create a bit of overlap in the chart between step t and steps t-1
# and t + 1
OVERLAP_OFFSET = 0.1
df[BORROW_DURATION] = df[END_T] - df[START_T] + OVERLAP_OFFSET
df[USAGE] = df.apply(
    lambda r: (
        f"{r[TENSOR_SIZE] / r[BUFFER_SIZE] * 100:.2f}%"
        if BUFFER_SIZE in r and pd.notna(r[BUFFER_SIZE]) and r[BUFFER_SIZE] != 0
        else "N.A."
    ),
    axis=1,
)

df[LABEL] = df.apply(
    lambda r: f"Size={r[TENSOR_SIZE]}{r[BUFFER_TYPE]} (Usage: {r[USAGE]})",
    axis=1,
)

buffer_ids = sorted(df[BUFFER_ID].unique())
N_BUFFERS = len(buffer_ids)
colors_array = pc.sample_colorscale(
    "rainbow",
    [i / max(N_BUFFERS - 1, 1) for i in range(N_BUFFERS)],
)
colors_by_buffer = dict(zip(buffer_ids, colors_array))

fig = make_subplots(
    rows=4,
    cols=4,
    specs=[
        [{"type": "xy", "colspan": 4, "rowspan": 2}, None, None, None],
        [None, None, None, None],
        [
            {"type": "table", "rowspan": 2, "colspan": 1},
            {"type": "table", "colspan": 3},
            None,
            None,
        ],
        [None, {"type": "table", "colspan": 3}, None, None],
    ],
    row_heights=[0.40, 0.40, 0.08, 0.12],
)

already_shown_legend = set({})

for row in df.itertuples():
    buffer_id = row.backing_buffer["id"]
    fig.add_trace(
        trace=go.Bar(
            base=[row.backing_buffer["start_borrow"]],
            x=[row.duration],
            y=[row.name],
            orientation="h",
            text=[row.label],
            textangle=0,
            marker={"color": colors_by_buffer[buffer_id]},
            name=f"{buffer_id}",
            legendgroup=buffer_id,
            showlegend=buffer_id not in already_shown_legend,
            hovertemplate=(
                "Borrow starts at %{base}\u003cbr\u003e"
                "Borrow"
                f" duration={row.duration - OVERLAP_OFFSET}\u003cbr\u003e"
                "Tensor name=%{y}\u003cbr\u003e%{text}\u003cbr\u003e"
                f"backing_buffer.id={buffer_id}"
            ),
            textposition="inside",
            insidetextanchor="middle",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    already_shown_legend.add(buffer_id)

buffers_df = (
    pd.DataFrame(
        df,
        columns=[
            BUFFER_ID,
            BUFFER_SIZE,
            BUFFER_TYPE,
        ],
    )
    .drop_duplicates()
    .sort_values("backing_buffer.id")
)

# print(oldfig.to_json())

fig.add_trace(
    go.Table(
        header=dict(values=["Buffer ID", "Size", "Type"]),
        cells=dict(
            values=[
                buffers_df[BUFFER_ID],
                buffers_df[BUFFER_SIZE],
                buffers_df[BUFFER_TYPE],
            ],
            align="left",
        ),
    ),
    row=3,
    col=1,
)

total_buffer_size = (
    df.drop_duplicates(BUFFER_ID)
    .apply(
        lambda r: r[BUFFER_SIZE] * (4 if r[BUFFER_TYPE] == "f32" else 1),
        axis=1,
    )
    .sum()
)

fig.add_trace(
    go.Table(
        header=dict(values=["Total size of buffers"]),
        cells=dict(
            values=[total_buffer_size],
            align="left",
        ),
    ),
    row=3,
    col=2,
)

EFFECTIVE_SIZE = "effective_size"
TOTAL_SIZE = "total_size"

df[EFFECTIVE_SIZE] = df[TENSOR_SIZE] * df[BUFFER_TYPE].map({"f32": 4, "u8": 1})

joined_along_lifetime_bounds = df.merge(
    df, left_on=END_T, right_on=START_T, suffixes=("_left", "_right")
)
joined_along_lifetime_bounds[TOTAL_SIZE] = (
    joined_along_lifetime_bounds[f"{EFFECTIVE_SIZE}_left"]
    + joined_along_lifetime_bounds[f"{EFFECTIVE_SIZE}_right"]
)
max_row = joined_along_lifetime_bounds.loc[
    joined_along_lifetime_bounds[TOTAL_SIZE].idxmax(),
    [
        f"{TENSOR_NAME}_left",
        TOTAL_SIZE,
    ],
]

max_row = max_row.rename({f"{TENSOR_NAME}_left": TENSOR_NAME})

fig.add_trace(
    go.Table(
        header=dict(values=["Peak memory usage", "Tensor"]),
        columnwidth=[30, 100],
        cells=dict(
            values=[
                max_row[TOTAL_SIZE],
                max_row[TENSOR_NAME].replace(";", " "),
            ],
            align="left",
        ),
    ),
    row=4,
    col=2,
)

max_x = (df[START_T] + df[BORROW_DURATION]).max()
min_y = df[TENSOR_NAME].iloc[0]
max_y = df[TENSOR_NAME].iloc[-1]

fig.update_yaxes(
    patch={
        "autorange": "reversed",
        "showticklabels": False,
        "minallowed": min_y,
        "maxallowed": max_y,
    }
)
fig.update_xaxes(
    patch={
        "title": "step",
        "minallowed": 1,
        "maxallowed": max_x,
    }
)
fig.update_layout({"legend": {"title": {"text": "Buffer ID"}}})

metadata_title = (
    f"Planner: {html.escape(str(planner))}"
    f" | Flags: {html.escape(str(flags))}"
)
if node_count is not None:
    metadata_title += f" | Nodes: {html.escape(str(node_count))}"

fig.update_layout(
    {
        "title": {
            "text": f"Static memory allocation<br><sup>{metadata_title}</sup>",
            "x": 0.5,
        },
        "margin": {"t": 90},
    }
)

print(fig.to_json())

fig.show()
