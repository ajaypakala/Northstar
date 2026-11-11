"""General-purpose, conservative CSV cleaning utilities for Northstar."""

import re
import pandas as pd

MISSING_VALUES = {
    "", "na", "n/a", "nan", "none", "null", "nil",
    "missing", "not available", "not applicable"
}

DATE_NAME_HINTS = {
    "date", "datetime", "timestamp", "time", "created_at",
    "updated_at", "birth_date", "joining_date"
}

ID_NAME_HINTS = {
    "id", "code", "zip", "postal", "phone", "mobile",
    "account_number", "employee_number", "roll_number",
    "registration_number"
}


def _tokens(column_name):
    return set(re.split(r"[_\W]+", str(column_name).lower()))


def is_identifier_column(column_name):
    """Recognize common identifier names, including orderid/customerid."""
    name = re.sub(r"[^a-z0-9]", "", str(column_name).lower())
    return bool(_tokens(column_name) & ID_NAME_HINTS) or any(
        name.endswith(suffix) for suffix in ("id", "code", "number", "phone", "mobile", "zipcode", "postalcode")
    )


def is_date_named_column(column_name):
    return bool(_tokens(column_name) & DATE_NAME_HINTS) or str(column_name).lower().endswith(("date", "datetime", "timestamp"))


def clean_column_names(df):
    df = df.copy()
    seen = {}
    names = []
    for column in df.columns:
        name = re.sub(r"[^\w]+", "_", str(column).strip().lower())
        name = re.sub(r"_+", "_", name).strip("_") or "unnamed_column"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count + 1}")
    df.columns = names
    return df


def standardize_text(df):
    df = df.copy()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        values = values.mask(values.str.lower().isin(MISSING_VALUES), pd.NA)
        df[column] = values
    return df


def parse_numeric_series(series):
    """Parse common numeric strings; preserve parentheses as negatives."""
    text = series.astype("string").str.strip()
    parenthesized = text.str.match(r"^\(.*\)$", na=False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    text = text.str.replace(r"[₹$€£,\s%]", "", regex=True)
    values = pd.to_numeric(text, errors="coerce")
    values.loc[parenthesized & values.notna()] *= -1
    return values


def detect_numeric_columns(df, threshold=0.95):
    detected = []
    for column in df.columns:
        if is_identifier_column(column) or is_date_named_column(column):
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            detected.append(column)
            continue
        if not (pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)):
            continue
        non_missing = series.dropna()
        if non_missing.empty:
            continue
        converted = parse_numeric_series(non_missing)
        if converted.notna().mean() >= threshold:
            detected.append(column)
    return detected


def parse_dates(series):
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _looks_like_date(series):
    sample = series.dropna().astype("string").head(100)
    if sample.empty:
        return False
    # Avoid converting plain numeric strings such as 2024 or 12 into dates.
    mask = sample.str.contains(r"[-/:]|[A-Za-z]{3,}", regex=True, na=False)
    return bool(mask.mean() >= 0.5)


def detect_datetime_columns(df, threshold=0.80):
    detected = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            detected.append(column)
            continue
        if is_identifier_column(column):
            continue
        if not (pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)):
            continue
        values = series.dropna()
        if values.empty:
            continue
        if not is_date_named_column(column) and not _looks_like_date(values):
            continue
        parsed = parse_dates(values)
        if parsed.notna().mean() >= threshold:
            detected.append(column)
    return detected


def fill_missing_values(df, numeric_strategy="preserve", categorical_strategy="preserve"):
    """Optional imputation. Identifiers and dates are never imputed."""
    valid_numeric = {"preserve", "median", "mean"}
    valid_categorical = {"preserve", "mode", "unknown"}
    if numeric_strategy not in valid_numeric:
        raise ValueError(f"numeric_strategy must be one of {sorted(valid_numeric)}")
    if categorical_strategy not in valid_categorical:
        raise ValueError(f"categorical_strategy must be one of {sorted(valid_categorical)}")

    result = df.copy()
    for column in result.columns:
        if is_identifier_column(column) or pd.api.types.is_datetime64_any_dtype(result[column]):
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            if numeric_strategy == "median":
                fill_value = result[column].median()
            elif numeric_strategy == "mean":
                fill_value = result[column].mean()
            else:
                continue
            if pd.notna(fill_value):
                result[column] = result[column].fillna(fill_value)
        elif categorical_strategy == "unknown":
            result[column] = result[column].fillna("Unknown")
        elif categorical_strategy == "mode":
            modes = result[column].mode(dropna=True)
            if not modes.empty:
                result[column] = result[column].fillna(modes.iloc[0])
    return result


def detect_outliers(df):
    """Count IQR outliers; never delete or alter them."""
    report = {}
    for column in df.select_dtypes(include="number").columns:
        if is_identifier_column(column):
            continue
        values = df[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            report[column] = 0
            continue
        count = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        report[column] = int(count)
    return report


def detect_negative_values(df):
    report = {}
    for column in df.select_dtypes(include="number").columns:
        count = int(df[column].lt(0).sum())
        if count:
            report[column] = count
    return report


def detect_repeated_identifier_values(df):
    report = {}
    for column in df.columns:
        if is_identifier_column(column):
            values = df[column].dropna()
            count = int(values.duplicated().sum())
            if count:
                report[column] = count
    return report


def clean_dataset(
    df,
    numeric_strategy="preserve",
    categorical_strategy="preserve",
    remove_exact_duplicates=True,
):
    """Clean safe formatting issues without guessing business-specific rules."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned = clean_column_names(df)
    cleaned = standardize_text(cleaned)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if remove_exact_duplicates:
        cleaned = cleaned.drop_duplicates().copy()

    for column in detect_datetime_columns(cleaned):
        cleaned[column] = parse_dates(cleaned[column])

    for column in detect_numeric_columns(cleaned):
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = parse_numeric_series(cleaned[column])

    cleaned = fill_missing_values(
        cleaned,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )
    return cleaned


def cleaning_report(original_df, cleaned_df):
    original_missing = int(original_df.isna().sum().sum())
    cleaned_missing = int(cleaned_df.isna().sum().sum())
    return {
        "original_rows": int(len(original_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "rows_removed": int(len(original_df) - len(cleaned_df)),
        "original_columns": int(len(original_df.columns)),
        "cleaned_columns": int(len(cleaned_df.columns)),
        "columns_removed": int(len(original_df.columns) - len(cleaned_df.columns)),
        "exact_duplicates_before": int(original_df.duplicated().sum()),
        "exact_duplicates_after": int(cleaned_df.duplicated().sum()),
        "missing_values_before": original_missing,
        "missing_values_after": cleaned_missing,
        "missing_values_by_column": {
            str(c): int(n) for c, n in cleaned_df.isna().sum().items() if n > 0
        },
        "detected_numeric_columns": [str(c) for c in cleaned_df.select_dtypes(include="number").columns],
        "detected_datetime_columns": [
            str(c) for c in cleaned_df.columns
            if pd.api.types.is_datetime64_any_dtype(cleaned_df[c])
        ],
        "detected_categorical_columns": [
            str(c) for c in cleaned_df.select_dtypes(include=["object", "string", "category"]).columns
        ],
        "negative_values": detect_negative_values(cleaned_df),
        "repeated_identifier_values": detect_repeated_identifier_values(cleaned_df),
        "outliers": detect_outliers(cleaned_df),
    }
