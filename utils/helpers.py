"""General utility helpers used by the Northstar Streamlit application."""

from datetime import datetime
import re
import pandas as pd


def get_numeric_columns(df):
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()


def get_datetime_columns(df):
    return [
        column for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]


def preview_dataframe(df, rows=10):
    rows = max(0, int(rows))
    return df.head(rows).copy()


def generate_dataset_summary(df, max_columns=60, max_categories=10):
    """Create a compact, JSON-safe dataset description for the AI service."""
    details = []
    for column in df.columns[:max_columns]:
        series = df[column]
        item = {
            "column": str(column),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            item["statistics"] = {
                "min": _safe_number(series.min()),
                "max": _safe_number(series.max()),
                "mean": _safe_number(series.mean()),
                "median": _safe_number(series.median()),
            }
        else:
            item["top_values"] = {
                str(k): int(v)
                for k, v in series.value_counts(dropna=True).head(max_categories).items()
            }
        details.append(item)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "column_details": details,
    }
    return summary


def _safe_number(value):
    if pd.isna(value):
        return None
    return float(value)


def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def generate_report_filename(prefix="northstar_report"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "report"
    return f"{safe_prefix}_{timestamp}.pdf"
