
# analysis/cleaner.py

import re
import json
import pandas as pd
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MISSING_VALUES = {
    "", "na", "n/a", "nan", "none", "null",
    "nil", "missing", "not available"
}

DATE_NAME_HINTS = {
    "date", "datetime", "timestamp", "time",
    "created_at", "updated_at", "birth_date",
    "joining_date"
}

ID_NAME_HINTS = {
    "id", "code", "zip", "postal", "phone",
    "mobile", "account_number", "employee_number",
    "roll_number", "registration_number"
}

# These symbols are removed when parsing numeric strings.
# Percent signs are removed without changing the numeric scale.
NUMERIC_CLEAN_PATTERN = re.compile(r"[₹$€£,\s%]")


# --------------------------------------------------
# Column names
# --------------------------------------------------

def clean_column_names(df):
    """Standardize column names and make duplicate names unique."""
    df = df.copy()
    new_columns = []
    seen = {}

    for column in df.columns:
        name = str(column).strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "unnamed_column"

        count = seen.get(name, 0)
        seen[name] = count + 1

        if count:
            name = f"{name}_{count + 1}"

        new_columns.append(name)

    df.columns = new_columns
    return df


def is_identifier_column(column_name):
    """Identify likely identifier columns using complete name tokens."""
    tokens = set(re.split(r"[_\W]+", str(column_name).lower()))
    return bool(tokens.intersection(ID_NAME_HINTS))


def is_date_named_column(column_name):
    """Check date hints using complete name tokens."""
    tokens = set(re.split(r"[_\W]+", str(column_name).lower()))
    return bool(tokens.intersection(DATE_NAME_HINTS))


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def standardize_text(df):
    """Trim whitespace and standardize common missing markers."""
    df = df.copy()

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].astype("string").str.strip()
        values = values.str.replace(r"\s+", " ", regex=True)

        lower_values = values.str.lower()
        values = values.mask(lower_values.isin(MISSING_VALUES), pd.NA)

        df[column] = values

    return df


# --------------------------------------------------
# Numeric detection
# --------------------------------------------------

def parse_numeric_series(series):
    """Convert common formatted numeric strings into numbers."""
    text = series.astype("string").str.strip()

    # Support values such as (120.50) to represent -120.50.
    is_parenthesized = text.str.match(r"^\(.*\)$", na=False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    text = text.str.replace(NUMERIC_CLEAN_PATTERN, "", regex=True)
    numeric = pd.to_numeric(text, errors="coerce")

    numeric.loc[is_parenthesized & numeric.notna()] *= -1
    return numeric


def detect_numeric_columns(df, threshold=0.95):
    """Detect numeric-looking text columns without converting likely IDs."""
    detected = []

    for column in df.columns:
        if is_identifier_column(column):
            continue

        if is_date_named_column(column):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            detected.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.StringDtype)
        ):
            continue

        non_missing = df[column].dropna()

        if len(non_missing) == 0:
            continue

        converted = parse_numeric_series(non_missing)
        valid_ratio = converted.notna().mean()

        if valid_ratio >= threshold:
            detected.append(column)

    return detected


# --------------------------------------------------
# Date detection
# --------------------------------------------------

def parse_dates(series):
    """Parse mixed date formats where supported by the installed pandas."""
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def looks_like_date_values(series):
    """Require date-like characters to reduce accidental number conversion."""
    sample = series.dropna().astype("string").head(100)

    if sample.empty:
        return False

    date_pattern = sample.str.contains(
        r"[-/:]|[A-Za-z]{3,}",
        regex=True,
        na=False
    )

    return date_pattern.mean() >= 0.5


def detect_datetime_columns(df, threshold=0.80):
    """Detect date columns using column names and actual values."""
    detected = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            detected.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.StringDtype)
        ):
            continue

        non_missing = series.dropna()

        if len(non_missing) == 0:
            continue

        # Do not interpret ID-like values as dates.
        if is_identifier_column(column):
            continue

        # For columns without a date-like name, require date-like values.
        if not is_date_named_column(column):
            if not looks_like_date_values(non_missing):
                continue

        parsed = parse_dates(non_missing)
        valid_ratio = parsed.notna().mean()

        if valid_ratio >= threshold:
            detected.append(column)

    return detected


# --------------------------------------------------
# Missing values
# --------------------------------------------------

def fill_missing_values(
    df,
    numeric_strategy="preserve",
    categorical_strategy="preserve"
):
    """
    Fill missing values only when explicitly requested.

    numeric_strategy:
        preserve, median, mean

    categorical_strategy:
        preserve, mode, unknown
    """
    df = df.copy()

    for column in df.columns:
        if is_identifier_column(column):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if numeric_strategy == "median":
                value = df[column].median()
                if pd.notna(value):
                    df[column] = df[column].fillna(value)

            elif numeric_strategy == "mean":
                value = df[column].mean()
                if pd.notna(value):
                    df[column] = df[column].fillna(value)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            # Dates are preserved unless a separate date policy is defined.
            continue

        elif numeric_strategy or categorical_strategy:
            if categorical_strategy == "unknown":
                df[column] = df[column].fillna("Unknown")

            elif categorical_strategy == "mode":
                modes = df[column].mode(dropna=True)
                if not modes.empty:
                    df[column] = df[column].fillna(modes.iloc[0])

    return df


# --------------------------------------------------
# Outlier and quality checks
# --------------------------------------------------

def detect_outliers(df):
    """Flag IQR outliers; do not remove or modify them."""
    results = {}

    for column in df.select_dtypes(include="number").columns:
        if is_identifier_column(column):
            continue

        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            results[column] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        results[column] = int(
            ((values < lower) | (values > upper)).sum()
        )

    return results


def detect_negative_values(df):
    """Report negative values in numeric columns without assuming they are invalid."""
    results = {}

    for column in df.select_dtypes(include="number").columns:
        count = int((df[column] < 0).sum())

        if count:
            results[column] = count

    return results


def detect_repeated_identifier_values(df):
    """
    Report repeated values in likely ID columns.
    Repeated IDs are not automatically deleted because they may be valid.
    """
    results = {}

    for column in df.columns:
        if not is_identifier_column(column):
            continue

        values = df[column].dropna()
        duplicate_count = int(values.duplicated().sum())

        if duplicate_count:
            results[column] = duplicate_count

    return results


# --------------------------------------------------
# Main cleaning pipeline
# --------------------------------------------------

def clean_dataset(
    df,
    numeric_strategy="preserve",
    categorical_strategy="preserve",
    remove_exact_duplicates=True
):
    """
    General-purpose CSV cleaning pipeline.

    Safe defaults:
    - No dataset-specific assumptions.
    - No automatic deletion of repeated IDs.
    - No automatic removal of negative values or outliers.
    - No automatic filling of missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned = df.copy()

    # 1. Standardize column names.
    cleaned = clean_column_names(cleaned)

    # 2. Standardize text and missing markers.
    cleaned = standardize_text(cleaned)

    # 3. Remove completely empty rows and columns.
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.dropna(axis=1, how="all")

    # 4. Remove exact duplicate rows only.
    if remove_exact_duplicates:
        cleaned = cleaned.drop_duplicates()

    # 5. Detect and convert date columns.
    date_columns = detect_datetime_columns(cleaned)

    for column in date_columns:
        cleaned[column] = parse_dates(cleaned[column])

    # 6. Detect and convert numeric-looking text columns.
    numeric_text_columns = detect_numeric_columns(cleaned)

    for column in numeric_text_columns:
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = parse_numeric_series(cleaned[column])

    # 7. Apply optional missing-value strategies.
    cleaned = fill_missing_values(
        cleaned,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy
    )

    return cleaned


# --------------------------------------------------
# Cleaning report
# --------------------------------------------------

def cleaning_report(original_df, cleaned_df):
    """Create a report that compares the input and cleaned datasets."""
    original = original_df.copy()
    cleaned = cleaned_df.copy()

    original_missing = int(original.isna().sum().sum())
    cleaned_missing = int(cleaned.isna().sum().sum())

    report = {
        "original_rows": int(len(original)),
        "cleaned_rows": int(len(cleaned)),
        "rows_removed": int(len(original) - len(cleaned)),

        "original_columns": int(len(original.columns)),
        "cleaned_columns": int(len(cleaned.columns)),
        "columns_removed": int(
            len(original.columns) - len(cleaned.columns)
        ),

        "exact_duplicates_before": int(original.duplicated().sum()),
        "exact_duplicates_after": int(cleaned.duplicated().sum()),

        "missing_values_before": original_missing,
        "missing_values_after": cleaned_missing,

        "missing_values_by_column": {
            str(column): int(count)
            for column, count in cleaned.isna().sum().items()
            if count > 0
        },

        "detected_numeric_columns": [
            str(column)
            for column in cleaned.select_dtypes(include="number").columns
        ],

        "detected_datetime_columns": [
            str(column)
            for column in cleaned.columns
            if pd.api.types.is_datetime64_any_dtype(cleaned[column])
        ],

        "detected_categorical_columns": [
            str(column)
            for column in cleaned.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        ],

        "negative_values": detect_negative_values(cleaned),
        "repeated_identifier_values": detect_repeated_identifier_values(cleaned),
        "outliers": detect_outliers(cleaned)
    }

    return report


def get_quality_summary(df):
    """Return a compact quality summary for the current dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "negative_values": detect_negative_values(df),
        "repeated_identifier_values": detect_repeated_identifier_values(df),
        "outliers": detect_outliers(df)
    }
