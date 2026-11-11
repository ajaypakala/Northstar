
"""
Northstar
Universal AI Data Cleaning Engine

Designed to work with different CSV datasets:
Sales, HR, Finance, Healthcare, Education,
Marketing, Inventory and more.

No dataset-specific business rules are applied automatically.
"""

import re
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

DATE_KEYWORDS = (
    "date", "time", "timestamp", "datetime",
    "created_at", "updated_at", "dob",
    "birth_date", "joining_date"
)

IDENTIFIER_KEYWORDS = (
    "id", "code", "zip", "postal",
    "phone", "mobile", "account_number",
    "employee_number", "roll_number"
)

MISSING_TEXT_VALUES = {
    "", "null", "none", "nan", "n/a", "na",
    "not available", "not applicable", "-"
}


# ==========================================================
# 1. CLEAN COLUMN NAMES
# ==========================================================

def clean_column_names(df):
    """
    Standardize column names and handle duplicate names.
    """

    df = df.copy()

    columns = []

    for column in df.columns:

        name = str(column).strip().lower()

        name = re.sub(r"[^\w]+", "_", name)

        name = name.strip("_")

        if not name:
            name = "unnamed_column"

        columns.append(name)

    counts = {}
    unique_columns = []

    for column in columns:

        counts[column] = counts.get(column, 0) + 1

        if counts[column] == 1:
            unique_columns.append(column)
        else:
            unique_columns.append(
                f"{column}_{counts[column]}"
            )

    df.columns = unique_columns

    return df


# ==========================================================
# 2. STANDARDIZE TEXT VALUES
# ==========================================================

def standardize_text(df):
    """
    Remove extra whitespace and normalize common
    representations of missing text values.
    """

    df = df.copy()

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):

            values = df[column].astype("string")

            values = (
                values
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            # Normalize common missing-value indicators
            missing_mask = (
                values.str.lower()
                .isin(MISSING_TEXT_VALUES)
            )

            values = values.mask(missing_mask, pd.NA)

            df[column] = values

    return df


# ==========================================================
# 3. REMOVE EMPTY ROWS
# ==========================================================

def remove_empty_rows(df):
    """
    Remove rows where all values are missing.
    """

    return df.dropna(
        how="all"
    ).reset_index(drop=True)


# ==========================================================
# 4. REMOVE EMPTY COLUMNS
# ==========================================================

def remove_empty_columns(df):
    """
    Remove columns where all values are missing.
    """

    return df.dropna(
        axis=1,
        how="all"
    )


# ==========================================================
# 5. REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(df, subset=None):
    """
    Remove duplicate records.

    Default:
        Removes only completely identical rows.

    Optional:
        subset can contain selected business key columns.
    """

    df = df.copy()

    if subset is not None:

        if isinstance(subset, str):
            subset = [subset]

        missing = [
            col for col in subset
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Columns not found: {missing}"
            )

    return df.drop_duplicates(
        subset=subset,
        keep="first"
    ).reset_index(drop=True)


# ==========================================================
# 6. DETECT DATETIME COLUMNS
# ==========================================================

def detect_datetime_columns(df, threshold=0.95):
    """
    Detect date columns using column names and values.

    Named date columns:
        Require at least 50% valid date values.

    Other columns:
        Require at least 95% valid date values.

    Invalid values become NaT.
    """

    df = df.copy()

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue

        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        original = df[column]

        non_null_count = original.notna().sum()

        if non_null_count == 0:
            continue

        name = column.lower()

        is_date_column = any(
            keyword in name
            for keyword in DATE_KEYWORDS
        )

        try:

            converted = pd.to_datetime(
                original,
                errors="coerce",
                format="mixed"
            )

            valid_count = converted.notna().sum()

            valid_percentage = (
                valid_count / non_null_count
            )

            required_threshold = (
                0.50 if is_date_column else threshold
            )

            if valid_percentage >= required_threshold:

                df[column] = converted

        except (ValueError, TypeError, OverflowError):
            continue

    return df


# ==========================================================
# 7. DETECT NUMERIC COLUMNS
# ==========================================================

def detect_numeric_columns(df, threshold=0.95):
    """
    Convert numeric-looking text columns into numeric types.

    Avoids automatically converting identifiers.
    """

    df = df.copy()

    for column in df.columns:

        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        name = column.lower()

        # Do not convert likely identifiers
        if any(
            keyword in name
            for keyword in IDENTIFIER_KEYWORDS
        ):
            continue

        # Do not convert date columns here
        if any(
            keyword in name
            for keyword in DATE_KEYWORDS
        ):
            continue

        original = df[column]

        non_null_count = original.notna().sum()

        if non_null_count == 0:
            continue

        cleaned = (
            original.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(r"^[₹$€£]\s*", "", regex=True)
            .str.replace("%", "", regex=False)
        )

        converted = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        valid_percentage = (
            converted.notna().sum() / non_null_count
        )

        if valid_percentage >= threshold:

            df[column] = converted

    return df


# ==========================================================
# 8. HANDLE MISSING VALUES
# ==========================================================

def fill_missing_values(
    df,
    numeric_strategy="median",
    categorical_strategy="mode",
    fill_datetime=False
):
    """
    Handle missing values according to column type.

    Numeric:
        median, mean or preserve.

    Categorical:
        mode, Unknown or preserve.

    Datetime:
        preserve by default.
    """

    df = df.copy()

    for column in df.columns:

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):

            if numeric_strategy == "median":

                value = df[column].median()

            elif numeric_strategy == "mean":

                value = df[column].mean()

            elif numeric_strategy == "preserve":

                continue

            else:

                raise ValueError(
                    "Invalid numeric strategy."
                )

            if pd.notna(value):

                df[column] = df[column].fillna(value)

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[column]):

            if fill_datetime:

                df[column] = (
                    df[column]
                    .ffill()
                    .bfill()
                )

        # Categorical columns
        else:

            if categorical_strategy == "mode":

                modes = df[column].mode(dropna=True)

                if not modes.empty:

                    df[column] = df[column].fillna(
                        modes.iloc[0]
                    )

            elif categorical_strategy == "unknown":

                df[column] = df[column].fillna("Unknown")

            elif categorical_strategy == "preserve":

                continue

            else:

                raise ValueError(
                    "Invalid categorical strategy."
                )

    return df


# ==========================================================
# 9. DETECT OUTLIERS
# ==========================================================

def detect_outliers(df):
    """
    Identify numerical outliers using IQR.

    Does not remove outliers automatically.
    """

    results = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        results.append({
            "Column": column,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outlier Count": count
        })

    return pd.DataFrame(
        results,
        columns=[
            "Column",
            "Lower Bound",
            "Upper Bound",
            "Outlier Count"
        ]
    )


# ==========================================================
# 10. VALIDATE DATA TYPES
# ==========================================================

def validate_data_types(df):
    """
    Classify every column.
    """

    results = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            dtype = "Datetime"

        elif pd.api.types.is_numeric_dtype(df[column]):
            dtype = "Numeric"

        else:
            dtype = "Categorical"

        results.append({
            "Column": column,
            "Data Type": dtype,
            "Pandas Type": str(df[column].dtype)
        })

    return pd.DataFrame(results)


# ==========================================================
# 11. COMPLETE CLEANING PIPELINE
# ==========================================================

def clean_dataset(
    df,
    duplicate_subset=None,
    numeric_strategy="median",
    categorical_strategy="mode",
    fill_datetime=False
):
    """
    Universal CSV cleaning pipeline.

    No industry-specific assumptions are applied.

    Parameters:
        df: Input DataFrame.
        duplicate_subset: Optional duplicate key columns.
        numeric_strategy: median, mean or preserve.
        categorical_strategy: mode, unknown or preserve.
        fill_datetime: False by default.

    Returns:
        Cleaned DataFrame.
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if df.empty:

        raise ValueError(
            "Uploaded dataset is empty."
        )

    cleaned = df.copy()

    # Step 1: Standardize column names
    cleaned = clean_column_names(cleaned)

    # Step 2: Standardize text values
    cleaned = standardize_text(cleaned)

    # Step 3: Remove completely empty rows
    cleaned = remove_empty_rows(cleaned)

    # Step 4: Remove completely empty columns
    cleaned = remove_empty_columns(cleaned)

    # Step 5: Remove exact duplicate rows
    cleaned = remove_duplicates(
        cleaned
    )

    # Step 6: Optional business-key duplicates
    if duplicate_subset is not None:

        cleaned = remove_duplicates(
            cleaned,
            subset=duplicate_subset
        )

    # Step 7: Detect datetime columns
    cleaned = detect_datetime_columns(cleaned)

    # Step 8: Detect numeric columns
    cleaned = detect_numeric_columns(cleaned)

    # Step 9: Handle missing values
    cleaned = fill_missing_values(
        cleaned,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        fill_datetime=fill_datetime
    )

    # Step 10: Reset index
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


# ==========================================================
# 12. CLEANING REPORT
# ==========================================================

def cleaning_report(original_df, cleaned_df):
    """
    Generate detailed before-and-after statistics.
    """

    original_missing = int(
        original_df.isnull().sum().sum()
    )

    cleaned_missing = int(
        cleaned_df.isnull().sum().sum()
    )

    type_summary = validate_data_types(cleaned_df)

    outlier_summary = detect_outliers(cleaned_df)

    report = {

        "Original Rows": int(len(original_df)),

        "Cleaned Rows": int(len(cleaned_df)),

        "Rows Removed": int(
            len(original_df) - len(cleaned_df)
        ),

        "Original Columns": int(
            original_df.shape[1]
        ),

        "Cleaned Columns": int(
            cleaned_df.shape[1]
        ),

        "Columns Removed": int(
            original_df.shape[1] - cleaned_df.shape[1]
        ),

        "Original Missing Values": original_missing,

        "Missing Values Remaining": cleaned_missing,

        "Missing Values Handled": max(
            0,
            original_missing - cleaned_missing
        ),

        "Original Exact Duplicates": int(
            original_df.duplicated().sum()
        ),

        "Duplicate Rows Remaining": int(
            cleaned_df.duplicated().sum()
        ),

        "Numeric Columns": int(
            (type_summary["Data Type"] == "Numeric").sum()
        ),

        "Categorical Columns": int(
            (type_summary["Data Type"] == "Categorical").sum()
        ),

        "Datetime Columns": int(
            (type_summary["Data Type"] == "Datetime").sum()
        ),

        "Detected Date Columns": type_summary.loc[
            type_summary["Data Type"] == "Datetime",
            "Column"
        ].tolist(),

        "Detected Numeric Columns": type_summary.loc[
            type_summary["Data Type"] == "Numeric",
            "Column"
        ].tolist(),

        "Detected Categorical Columns": type_summary.loc[
            type_summary["Data Type"] == "Categorical",
            "Column"
        ].tolist(),

        "Outlier Summary": outlier_summary.to_dict(
            orient="records"
        ),

        "Memory Usage (MB)": round(
            cleaned_df.memory_usage(deep=True).sum()
            / (1024 * 1024),
            2
        )

    }

    return report
