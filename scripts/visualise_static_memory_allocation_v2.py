#!/usr/bin/env python3
"""
Memory Allocation Visualizer - Redesigned
A clean, intuitive tool for analyzing static memory allocation plans.

Usage: python visualize_memory.py <static_memory_plan.json>
Dependencies: pip install pandas plotly numpy
"""

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BufferStats:
    """Statistics for a single buffer."""

    id: int
    element_type: str
    capacity_bytes: int
    peak_usage_bytes: int
    peak_usage_pct: float
    avg_utilization_pct: float
    tensor_count: int
    usage_timeline: List[int]


class MemoryAnalyzer:
    """Analyzes memory allocation data and generates visualizations."""

    BYTES_PER_F32 = 4
    BYTES_PER_U8 = 1
    COLORS = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#16a085",
        "#c0392b",
    ]

    def __init__(self, json_path: str):
        """Load and preprocess memory allocation data."""
        self.df = pd.read_json(json_path)
        self._preprocess_data()
        self.buffer_ids = sorted(self.df["buffer_id"].unique())
        self.max_time = int(self.df["end_time"].max()) + 1
        self.time_steps = np.arange(0, self.max_time)

    def _preprocess_data(self):
        """Extract and calculate derived fields."""
        # Normalize nested buffer data
        buffer_data = pd.json_normalize(self.df["backing_buffer"])
        buffer_data.columns = ["buffer_" + col for col in buffer_data.columns]
        self.df = self.df.join(buffer_data)

        # Rename for clarity
        self.df["start_time"] = self.df["buffer_start_borrow"]
        self.df["end_time"] = self.df["buffer_end_borrow"]
        self.df["duration"] = self.df["end_time"] - self.df["start_time"]

        # Calculate sizes in bytes
        self.df["bytes_per_element"] = self.df["buffer_element_type"].apply(
            lambda t: self.BYTES_PER_F32 if t == "f32" else self.BYTES_PER_U8
        )
        self.df["size_bytes"] = self.df["size"] * self.df["bytes_per_element"]
        self.df["buffer_capacity_bytes"] = (
            self.df["buffer_size"] * self.df["bytes_per_element"]
        )
        self.df["buffer_usage_pct"] = (
            self.df["size_bytes"] / self.df["buffer_capacity_bytes"] * 100
        )

    def get_buffer_color(self, buffer_id: int) -> str:
        """Get consistent color for a buffer."""
        return self.COLORS[buffer_id % len(self.COLORS)]

    def compute_buffer_stats(self) -> Dict[int, BufferStats]:
        """Compute comprehensive statistics for each buffer."""
        stats = {}

        for buffer_id in self.buffer_ids:
            buffer_df = self.df[self.df["buffer_id"] == buffer_id]
            capacity = buffer_df["buffer_capacity_bytes"].iloc[0]
            element_type = buffer_df["buffer_element_type"].iloc[0]

            # Calculate usage at each timestep
            usage_timeline = []
            for t in self.time_steps:
                active = buffer_df[
                    (buffer_df["start_time"] <= t) & (buffer_df["end_time"] > t)
                ]
                usage_timeline.append(active["size_bytes"].sum())

            peak_usage = max(usage_timeline)
            avg_utilization = np.mean(
                [u / capacity * 100 for u in usage_timeline if u > 0]
            )

            stats[buffer_id] = BufferStats(
                id=buffer_id,
                element_type=element_type,
                capacity_bytes=capacity,
                peak_usage_bytes=peak_usage,
                peak_usage_pct=peak_usage / capacity * 100 if capacity > 0 else 0,
                avg_utilization_pct=avg_utilization,
                tensor_count=len(buffer_df),
                usage_timeline=usage_timeline,
            )

        return stats

    def create_visualization(self) -> go.Figure:
        """Create comprehensive memory visualization dashboard."""
        buffer_stats = self.compute_buffer_stats()

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "table"}],
            ],
            subplot_titles=(
                "Memory Timeline - All Buffers",
                "Peak Usage by Buffer",
                "Buffer Utilization Over Time",
                "Tensor Size Distribution",
                "Summary Statistics",
            ),
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
        )

        # 1. Timeline visualization (Gantt-style)
        self._add_timeline_chart(fig, buffer_stats, row=1, col=1)

        # 2. Peak usage bar chart
        self._add_peak_usage_chart(fig, buffer_stats, row=2, col=1)

        # 3. Utilization over time
        self._add_utilization_chart(fig, buffer_stats, row=2, col=2)

        # 4. Size distribution
        self._add_size_distribution(fig, row=3, col=1)

        # 5. Summary table
        self._add_summary_table(fig, buffer_stats, row=3, col=2)

        # Layout configuration
        fig.update_layout(
            title={
                "text": "Static Memory Allocation Analysis",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24, "family": "Arial, sans-serif"},
            },
            height=1400,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            template="plotly_white",
            font=dict(size=11, family="Arial, sans-serif"),
        )

        return fig

    def _add_timeline_chart(
        self, fig: go.Figure, stats: Dict[int, BufferStats], row: int, col: int
    ):
        """Add Gantt-style timeline showing all tensor allocations."""
        shown_buffers = set()

        # Calculate bar width based on number of buffers
        bar_width = 0.8  # Make bars thicker

        for buffer_id in self.buffer_ids:
            buffer_df = self.df[self.df["buffer_id"] == buffer_id].sort_values(
                "start_time"
            )
            color = self.get_buffer_color(buffer_id)

            # Create horizontal bars for each tensor
            for idx, tensor in buffer_df.iterrows():
                opacity = 0.5 + (tensor["buffer_usage_pct"] / 100) * 0.4

                fig.add_trace(
                    go.Bar(
                        x=[tensor["duration"]],
                        y=[f"Buffer {buffer_id}"],
                        base=[tensor["start_time"]],
                        orientation="h",
                        name=f"Buffer {buffer_id}",
                        legendgroup=f"buffer_{buffer_id}",
                        showlegend=buffer_id not in shown_buffers,
                        width=bar_width,  # Set explicit bar width
                        marker=dict(
                            color=color,
                            opacity=opacity,
                            line=dict(color="white", width=1),
                        ),
                        customdata=[
                            [
                                tensor["name"],
                                tensor["size_bytes"],
                                tensor["buffer_usage_pct"],
                                tensor["start_time"],
                                tensor["end_time"],
                            ]
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Time: [%{customdata[3]} → %{customdata[4]}]<br>"
                            "Size: %{customdata[1]:,} bytes<br>"
                            "Buffer Usage: %{customdata[2]:.1f}%<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                shown_buffers.add(buffer_id)

        fig.update_xaxes(title_text="Time Step", row=row, col=col)
        fig.update_yaxes(title_text="Buffer", row=row, col=col)

    def _add_peak_usage_chart(
        self, fig: go.Figure, stats: Dict[int, BufferStats], row: int, col: int
    ):
        """Add bar chart comparing peak usage across buffers."""
        buffer_labels = [f"Buffer {bid}" for bid in self.buffer_ids]
        peak_usage = [stats[bid].peak_usage_bytes for bid in self.buffer_ids]
        capacities = [stats[bid].capacity_bytes for bid in self.buffer_ids]
        colors = [self.get_buffer_color(bid) for bid in self.buffer_ids]

        # Peak usage bars
        fig.add_trace(
            go.Bar(
                x=buffer_labels,
                y=peak_usage,
                name="Peak Usage",
                marker_color=colors,
                text=[f"{p/c*100:.1f}%" for p, c in zip(peak_usage, capacities)],
                textposition="outside",
                hovertemplate="%{x}<br>Peak: %{y:,} bytes<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Capacity reference line
        fig.add_trace(
            go.Scatter(
                x=buffer_labels,
                y=capacities,
                mode="markers",
                name="Capacity",
                marker=dict(symbol="diamond", size=12, color="black"),
                hovertemplate="Capacity: %{y:,} bytes<extra></extra>",
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(title_text="Bytes", row=row, col=col)

    def _add_utilization_chart(
        self, fig: go.Figure, stats: Dict[int, BufferStats], row: int, col: int
    ):
        """Add line chart showing buffer utilization over time."""
        for buffer_id in self.buffer_ids:
            stat = stats[buffer_id]
            utilization_pct = [
                u / stat.capacity_bytes * 100 for u in stat.usage_timeline
            ]

            fig.add_trace(
                go.Scatter(
                    x=self.time_steps,
                    y=utilization_pct,
                    name=f"Buffer {buffer_id}",
                    legendgroup=f"buffer_{buffer_id}",
                    showlegend=False,
                    mode="lines",
                    line=dict(width=2, color=self.get_buffer_color(buffer_id)),
                    hovertemplate=f"Step %{{x}}<br>Utilization: %{{y:.1f}}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # Add 100% reference
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="100% Capacity",
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Time Step", row=row, col=col)
        fig.update_yaxes(title_text="Utilization (%)", range=[0, 105], row=row, col=col)

    def _add_size_distribution(self, fig: go.Figure, row: int, col: int):
        """Add box plot showing tensor size distributions per buffer."""
        for buffer_id in self.buffer_ids:
            buffer_df = self.df[self.df["buffer_id"] == buffer_id]

            fig.add_trace(
                go.Box(
                    y=buffer_df["size_bytes"],
                    name=f"Buffer {buffer_id}",
                    legendgroup=f"buffer_{buffer_id}",
                    showlegend=False,
                    marker_color=self.get_buffer_color(buffer_id),
                    boxmean="sd",
                    hovertemplate="Size: %{y:,} bytes<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.update_yaxes(title_text="Tensor Size (bytes)", type="log", row=row, col=col)

    def _add_summary_table(
        self, fig: go.Figure, stats: Dict[int, BufferStats], row: int, col: int
    ):
        """Add summary statistics table."""
        # Calculate overall stats
        total_capacity = sum(s.capacity_bytes for s in stats.values())
        total_usage_timeline = [
            sum(stats[bid].usage_timeline[t] for bid in self.buffer_ids)
            for t in range(self.max_time)
        ]
        peak_total = max(total_usage_timeline)
        avg_total = np.mean([u for u in total_usage_timeline if u > 0])

        # Build table data
        headers = ["Metric", "Value"]
        rows = [
            ["<b>Overall</b>", ""],
            ["Total Capacity", f"{total_capacity:,} bytes"],
            [
                "Peak Usage",
                f"{peak_total:,} bytes ({peak_total/total_capacity*100:.1f}%)",
            ],
            [
                "Avg Usage",
                f"{avg_total:,.0f} bytes ({avg_total/total_capacity*100:.1f}%)",
            ],
            ["Total Tensors", str(len(self.df))],
            ["Duration", f"{self.max_time} steps"],
            ["", ""],
        ]

        for buffer_id in self.buffer_ids:
            stat = stats[buffer_id]
            rows.extend(
                [
                    [f"<b>Buffer {buffer_id} ({stat.element_type})</b>", ""],
                    ["  Capacity", f"{stat.capacity_bytes:,} bytes"],
                    ["  Peak Usage", f"{stat.peak_usage_pct:.1f}%"],
                    ["  Avg Utilization", f"{stat.avg_utilization_pct:.1f}%"],
                    ["  Tensors", str(stat.tensor_count)],
                ]
            )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#34495e",
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[[r[0] for r in rows], [r[1] for r in rows]],
                    fill_color=[
                        ["#ecf0f1" if i % 2 else "white" for i in range(len(rows))]
                    ],
                    align="left",
                    font=dict(size=11),
                    height=24,
                ),
            ),
            row=row,
            col=col,
        )

    def print_summary(self):
        """Print text summary to console."""
        stats = self.compute_buffer_stats()

        print("=" * 70)
        print("MEMORY ALLOCATION SUMMARY")
        print("=" * 70)

        total_capacity = sum(s.capacity_bytes for s in stats.values())
        print(f"\nTotal Capacity: {total_capacity:,} bytes")
        print(f"Number of Buffers: {len(self.buffer_ids)}")
        print(f"Total Tensors: {len(self.df)}")
        print(f"Timeline Duration: {self.max_time} steps")

        print("\nPer-Buffer Analysis:")
        for buffer_id in self.buffer_ids:
            stat = stats[buffer_id]
            print(f"\n  Buffer {buffer_id} ({stat.element_type}):")
            print(f"    Capacity:        {stat.capacity_bytes:,} bytes")
            print(
                f"    Peak Usage:      {stat.peak_usage_bytes:,} bytes ({stat.peak_usage_pct:.1f}%)"
            )
            print(f"    Avg Utilization: {stat.avg_utilization_pct:.1f}%")
            print(f"    Tensor Count:    {stat.tensor_count}")

        print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_memory.py <static_memory_plan.json>")
        sys.exit(1)

    analyzer = MemoryAnalyzer(sys.argv[1])
    analyzer.print_summary()

    fig = analyzer.create_visualization()
    fig.show()


if __name__ == "__main__":
    main()
