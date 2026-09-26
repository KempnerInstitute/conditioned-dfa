"""Small analysis helpers for Info-Geo experiment reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictorScore:
    target: str
    predictor: str
    n: int
    pearson_r: float
    spearman_r: float
    r2: float


def predictor_scores(df: pd.DataFrame, *, target: str, predictors: list[str]) -> pd.DataFrame:
    """Correlate candidate predictors against one target."""

    rows = []
    for predictor in predictors:
        subset = df[[target, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.shape[0] < 3:
            rows.append(PredictorScore(target, predictor, int(subset.shape[0]), np.nan, np.nan, np.nan).__dict__)
            continue
        x = subset[predictor].to_numpy(dtype=float)
        y = subset[target].to_numpy(dtype=float)
        rows.append(
            PredictorScore(
                target=target,
                predictor=predictor,
                n=int(subset.shape[0]),
                pearson_r=safe_corr(x, y),
                spearman_r=safe_corr(rankdata(x), rankdata(y)),
                r2=simple_linear_r2(x, y),
            ).__dict__
        )
    return pd.DataFrame(rows)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with constant-vector protection."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def simple_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 for a one-predictor least-squares fit."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) <= 1e-12:
        return float("nan")
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform without requiring scipy."""

    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=float)
    sorted_x = x[order]
    start = 0
    while start < x.shape[0]:
        end = start + 1
        while end < x.shape[0] and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def write_markdown_report(path: str | Path, *, title: str, sections: list[tuple[str, str]]) -> None:
    """Write a compact markdown report."""

    path = Path(path)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.rstrip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, *, float_format: str = ".3f") -> str:
    """Render a DataFrame as markdown with stable float formatting."""

    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rendered = []
        for value in row:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, float) or isinstance(value, np.floating):
                rendered.append(f"{float(value):{float_format}}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
