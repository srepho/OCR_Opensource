"""Dashboard reporting: comparison tables and interactive charts for OCR benchmarks.

Generates summary tables, bar charts, scatter plots, radar charts, heatmaps,
and box plots comparing OCR model performance across multiple metrics.

Usage:
    from src.reporting.dashboard import generate_full_dashboard
    generate_full_dashboard(
        metrics_dir="results/metrics",
        outputs_dir="results/raw_outputs",
        output_dir="results/dashboard",
    )
"""

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(
    metrics_dir: str | Path,
    outputs_dir: str | Path,
) -> pd.DataFrame:
    """Load metrics and profile data for all models into a single DataFrame.

    Reads ``text_metrics.json`` from each model subdirectory under *metrics_dir*
    and ``profile.json`` from each model subdirectory under *outputs_dir*.

    Returns a DataFrame with one row per (model, page) combination, plus columns
    from the profile (avg_time_per_page, peak_gpu_memory_mb, pages_per_minute).
    Columns include:
        model, pdf_stem, page_num, content_type,
        ned, cer, wer, bleu, fuzzy_ratio,
        text_accuracy, composite,
        avg_time_per_page, median_time_per_page, pages_per_minute,
        peak_gpu_memory_mb, total_pages_profiled
    """
    metrics_dir = Path(metrics_dir)
    outputs_dir = Path(outputs_dir)

    rows: list[dict] = []

    for model_dir in sorted(metrics_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        text_metrics_path = model_dir / "text_metrics.json"
        if not text_metrics_path.exists():
            continue

        with open(text_metrics_path) as f:
            evaluation = json.load(f)

        model_key = evaluation.get("model", model_dir.name)

        # Load profile if available
        profile_path = outputs_dir / model_key / "profile.json"
        profile: dict = {}
        if profile_path.exists():
            with open(profile_path) as f:
                profile = json.load(f)

        for page in evaluation.get("per_page", []):
            tm = page.get("text_metrics", {})
            comp = page.get("composite", {})

            rows.append({
                "model": model_key,
                "pdf_stem": page.get("pdf_stem", ""),
                "page_num": page.get("page_num", 0),
                "content_type": page.get("content_type", "unknown"),
                # Text metrics
                "ned": tm.get("ned", np.nan),
                "cer": tm.get("cer", np.nan),
                "wer": tm.get("wer", np.nan),
                "bleu": tm.get("bleu", np.nan),
                "fuzzy_ratio": tm.get("fuzzy_ratio", np.nan),
                # Composite
                "text_accuracy": comp.get("text_accuracy", np.nan),
                "composite": comp.get("composite", np.nan),
                # Profile (same for every page of this model)
                "avg_time_per_page": profile.get("avg_time_per_page", np.nan),
                "median_time_per_page": profile.get("median_time_per_page", np.nan),
                "pages_per_minute": profile.get("pages_per_minute", np.nan),
                "peak_gpu_memory_mb": profile.get("peak_gpu_memory_mb", np.nan),
                "total_pages_profiled": profile.get("total_pages", 0),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary comparison table across all models.

    Returns a DataFrame with one row per model and columns:
        model, ned_mean, cer_mean, wer_mean, bleu_mean, fuzzy_ratio_mean,
        avg_time_per_page, peak_gpu_mb, composite_score
    Sorted by composite_score descending (best first).
    """
    if df.empty:
        return pd.DataFrame()

    metric_cols = ["ned", "cer", "wer", "bleu", "fuzzy_ratio", "composite"]
    agg_dict = {col: "mean" for col in metric_cols}

    summary = df.groupby("model").agg(agg_dict).reset_index()

    # Rename aggregated columns
    summary.rename(columns={
        "ned": "ned_mean",
        "cer": "cer_mean",
        "wer": "wer_mean",
        "bleu": "bleu_mean",
        "fuzzy_ratio": "fuzzy_ratio_mean",
        "composite": "composite_score",
    }, inplace=True)

    # Add profile columns (constant per model, take first value)
    profile_cols = ["avg_time_per_page", "peak_gpu_memory_mb"]
    profile_df = df.groupby("model")[profile_cols].first().reset_index()
    profile_df.rename(columns={"peak_gpu_memory_mb": "peak_gpu_mb"}, inplace=True)

    summary = summary.merge(profile_df, on="model", how="left")

    # Round for readability
    for col in ["ned_mean", "cer_mean", "wer_mean"]:
        summary[col] = summary[col].round(4)
    for col in ["bleu_mean", "fuzzy_ratio_mean", "composite_score"]:
        summary[col] = summary[col].round(2)
    summary["avg_time_per_page"] = summary["avg_time_per_page"].round(3)
    summary["peak_gpu_mb"] = summary["peak_gpu_mb"].round(1)

    summary.sort_values("composite_score", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)

    return summary


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

_COLOUR_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def plot_accuracy_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of text accuracy ``(1 - NED) * 100`` per model.

    Models are sorted by accuracy descending.
    """
    if df.empty:
        return go.Figure()

    model_acc = (
        df.groupby("model")["ned"]
        .mean()
        .reset_index()
    )
    model_acc["text_accuracy"] = (1 - model_acc["ned"]) * 100
    model_acc.sort_values("text_accuracy", ascending=True, inplace=True)

    fig = go.Figure(
        go.Bar(
            x=model_acc["text_accuracy"],
            y=model_acc["model"],
            orientation="h",
            marker_color=_COLOUR_PALETTE[: len(model_acc)],
            text=model_acc["text_accuracy"].round(1),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Text Accuracy by Model  ((1 - NED) x 100)",
        xaxis_title="Text Accuracy (%)",
        yaxis_title="Model",
        height=max(400, len(model_acc) * 35 + 120),
        margin=dict(l=180),
        template="plotly_white",
    )

    return fig


def plot_accuracy_vs_speed(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of text accuracy vs processing speed (pages/min).

    Each point represents a model.  Ideal is upper-right (high accuracy, fast).
    """
    if df.empty:
        return go.Figure()

    model_stats = (
        df.groupby("model")
        .agg(
            ned_mean=("ned", "mean"),
            pages_per_minute=("pages_per_minute", "first"),
        )
        .reset_index()
    )
    model_stats["text_accuracy"] = (1 - model_stats["ned_mean"]) * 100

    fig = go.Figure()

    for i, row in model_stats.iterrows():
        colour = _COLOUR_PALETTE[i % len(_COLOUR_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[row["pages_per_minute"]],
            y=[row["text_accuracy"]],
            mode="markers+text",
            marker=dict(size=14, color=colour),
            text=[row["model"]],
            textposition="top center",
            name=row["model"],
            showlegend=False,
        ))

    fig.update_layout(
        title="Accuracy vs Speed",
        xaxis_title="Speed (pages / minute)",
        yaxis_title="Text Accuracy (%)",
        template="plotly_white",
        height=550,
    )

    return fig


def plot_radar(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Radar chart comparing top *top_n* models across normalised metrics.

    Metrics displayed: text_accuracy, bleu (scaled 0-100), fuzzy_ratio,
    speed_score (min-max normalised pages_per_minute),
    memory_score (inverted peak_gpu so lower is better).
    """
    if df.empty:
        return go.Figure()

    model_stats = (
        df.groupby("model")
        .agg(
            ned_mean=("ned", "mean"),
            bleu_mean=("bleu", "mean"),
            fuzzy_mean=("fuzzy_ratio", "mean"),
            ppm=("pages_per_minute", "first"),
            gpu=("peak_gpu_memory_mb", "first"),
        )
        .reset_index()
    )
    model_stats["text_accuracy"] = (1 - model_stats["ned_mean"]) * 100

    # Select top N by text accuracy
    model_stats.sort_values("text_accuracy", ascending=False, inplace=True)
    top = model_stats.head(top_n).copy()

    # Normalise speed and memory to 0-100
    ppm_max = top["ppm"].max()
    top["speed_score"] = (top["ppm"] / ppm_max * 100) if ppm_max > 0 else 0

    gpu_max = top["gpu"].max()
    top["memory_score"] = ((1 - top["gpu"] / gpu_max) * 100) if gpu_max > 0 else 100

    categories = ["Text Accuracy", "BLEU", "Fuzzy Ratio", "Speed", "Memory Efficiency"]

    fig = go.Figure()

    for i, (_, row) in enumerate(top.iterrows()):
        values = [
            row["text_accuracy"],
            row["bleu_mean"],
            row["fuzzy_mean"],
            row["speed_score"],
            row["memory_score"],
        ]
        # Close the polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            name=row["model"],
            line_color=_COLOUR_PALETTE[i % len(_COLOUR_PALETTE)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Top {top_n} Models - Multi-Metric Radar",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_content_type_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of text accuracy per model per content type.

    Rows = models (sorted by overall accuracy), columns = content types.
    Cell values = mean text accuracy for that combination.
    """
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["text_accuracy"] = (1 - df["ned"]) * 100

    pivot = df.pivot_table(
        values="text_accuracy",
        index="model",
        columns="content_type",
        aggfunc="mean",
    )

    # Sort models by overall mean accuracy
    pivot["_overall"] = pivot.mean(axis=1)
    pivot.sort_values("_overall", ascending=False, inplace=True)
    pivot.drop(columns="_overall", inplace=True)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            colorbar_title="Accuracy (%)",
        )
    )

    fig.update_layout(
        title="Text Accuracy by Model and Content Type",
        xaxis_title="Content Type",
        yaxis_title="Model",
        height=max(400, len(pivot) * 30 + 150),
        margin=dict(l=180),
        template="plotly_white",
    )

    return fig


def plot_metric_distributions(df: pd.DataFrame) -> go.Figure:
    """Box plots of key metric distributions per model.

    Creates a subplot grid with one box-plot panel per metric:
    NED, CER, WER, BLEU, Fuzzy Ratio.
    """
    if df.empty:
        return go.Figure()

    metrics = [
        ("ned", "NED (lower is better)"),
        ("cer", "CER (lower is better)"),
        ("wer", "WER (lower is better)"),
        ("bleu", "BLEU (higher is better)"),
        ("fuzzy_ratio", "Fuzzy Ratio (higher is better)"),
    ]

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        subplot_titles=[label for _, label in metrics],
        vertical_spacing=0.06,
    )

    models = sorted(df["model"].unique())

    for row_idx, (col, _label) in enumerate(metrics, start=1):
        for i, model in enumerate(models):
            model_data = df.loc[df["model"] == model, col].dropna()
            fig.add_trace(
                go.Box(
                    y=model_data,
                    name=model,
                    marker_color=_COLOUR_PALETTE[i % len(_COLOUR_PALETTE)],
                    showlegend=(row_idx == 1),
                    legendgroup=model,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        title="Metric Distributions by Model",
        height=350 * len(metrics),
        template="plotly_white",
        boxmode="group",
    )

    return fig


# ---------------------------------------------------------------------------
# Full dashboard generation
# ---------------------------------------------------------------------------

def generate_full_dashboard(
    metrics_dir: str | Path,
    outputs_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate all dashboard artefacts and save them as HTML files.

    Saves the following into *output_dir*:
        - summary_table.csv
        - accuracy_bar.html
        - accuracy_vs_speed.html
        - radar.html
        - content_type_heatmap.html
        - metric_distributions.html
        - index.html  (links to all charts)

    Args:
        metrics_dir: Path to ``results/metrics`` (contains per-model dirs)
        outputs_dir: Path to ``results/raw_outputs`` (contains per-model dirs)
        output_dir: Path to write dashboard output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(metrics_dir, outputs_dir)
    if df.empty:
        print("No results found. Ensure metrics exist in the expected directories.")
        return

    n_models = df["model"].nunique()
    n_pages = len(df)
    print(f"Loaded {n_pages} page results across {n_models} models.")

    # Summary table
    summary = generate_summary_table(df)
    summary_csv_path = output_dir / "summary_table.csv"
    summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary table -> {summary_csv_path}")

    # Generate and save charts
    charts: list[tuple[str, str, go.Figure]] = [
        ("accuracy_bar.html", "Text Accuracy Bar Chart", plot_accuracy_bar(df)),
        ("accuracy_vs_speed.html", "Accuracy vs Speed", plot_accuracy_vs_speed(df)),
        ("radar.html", "Multi-Metric Radar (Top 5)", plot_radar(df, top_n=5)),
        ("content_type_heatmap.html", "Content Type Heatmap", plot_content_type_heatmap(df)),
        ("metric_distributions.html", "Metric Distributions", plot_metric_distributions(df)),
    ]

    chart_links: list[str] = []
    for filename, title, fig in charts:
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        chart_links.append(f'<li><a href="{filename}">{title}</a></li>')
        print(f"Saved {title} -> {path}")

    # Build index.html
    summary_html = summary.to_html(index=False, border=1, classes="summary-table")
    index_html = _build_index_html(summary_html, chart_links, n_models, n_pages)
    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Saved dashboard index -> {index_path}")


def _build_index_html(
    summary_html: str,
    chart_links: list[str],
    n_models: int,
    n_pages: int,
) -> str:
    """Build the dashboard index page HTML."""
    links_ul = "\n        ".join(chart_links)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Benchmark Dashboard</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px 40px;
            background: #f8f9fa; color: #212529;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats {{
            display: flex; gap: 20px; margin: 20px 0;
        }}
        .stat-card {{
            background: white; border-radius: 8px; padding: 20px 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
        }}
        .stat-card .number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-card .label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
        .summary-table {{
            border-collapse: collapse; width: 100%; background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px;
            overflow: hidden;
        }}
        .summary-table th {{
            background: #3498db; color: white; padding: 12px 16px;
            text-align: left; font-size: 0.85em; text-transform: uppercase;
        }}
        .summary-table td {{
            padding: 10px 16px; border-bottom: 1px solid #ecf0f1;
            font-size: 0.9em;
        }}
        .summary-table tr:hover {{ background: #f1f8ff; }}
        ul.chart-links {{
            list-style: none; padding: 0;
        }}
        ul.chart-links li {{
            margin: 10px 0;
        }}
        ul.chart-links a {{
            display: inline-block; padding: 10px 20px;
            background: #3498db; color: white; border-radius: 6px;
            text-decoration: none; font-weight: 500;
            transition: background 0.2s;
        }}
        ul.chart-links a:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <h1>OCR Benchmark Dashboard</h1>

    <div class="stats">
        <div class="stat-card">
            <div class="number">{n_models}</div>
            <div class="label">Models Compared</div>
        </div>
        <div class="stat-card">
            <div class="number">{n_pages}</div>
            <div class="label">Total Page Evaluations</div>
        </div>
    </div>

    <h2>Summary Table</h2>
    {summary_html}

    <h2>Interactive Charts</h2>
    <ul class="chart-links">
        {links_ul}
    </ul>

    <footer style="margin-top:40px; color:#95a5a6; font-size:0.8em;">
        Generated by OCR Benchmark Framework
    </footer>
</body>
</html>
"""
