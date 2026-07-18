# GAF Demo

Reads a time series from a CSV file and produces BMP visualizations of three
time-series-to-image transforms:

- **GASF** — Gramian Angular Summation Field
- **GADF** — Gramian Angular Difference Field
- **MTF** — Markov Transition Field (8 quantile bins)

## Build & Run

```bash
zig build gaf-demo -- <input.csv> [--split] [--colormap viridis|jet|grayscale]
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `<input.csv>` | required | Path to CSV file with the input time series |
| `--split` | off | Write 3 separate BMP files instead of one combined file |
| `--colormap` | `viridis` | Color mapping: `viridis`, `jet`, or `grayscale` |

## CSV Format

A single row of comma-separated f32 values:

```
0.1, 0.5, -0.3, 0.8, 0.2, -0.6, 1.0, 0.4
```

## Output

**Default (combined):**
```
output.bmp    ← GASF | GADF | MTF tiled horizontally with a 4px white border
```

**With `--split`:**
```
output_gasf.bmp
output_gadf.bmp
output_mtf.bmp
```

Output files are written to the current working directory.
