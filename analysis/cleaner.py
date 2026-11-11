
"""
Northstar
Advanced Data Cleaning Module

Features:
1. Column name standardization
2. Empty row removal
3. Empty column removal
4. Duplicate removal
5. Text standardization
6. Automatic datetime detection
7. Automatic numeric conversion
8. Missing value handling
9. Data type validation
10. Detailed cleaning report
"""

import re
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

DATE_KEYWORDS = [
    "date", "time", "timestamp",
    "created_at", "updated_at",
    "order_date", "delivery_date"
]

ID_KEYWORDS = [
    "id", "code", "zip", "postal",
    "phone", "mobile", "account_number"
]

CURRENCY_SYMBOLS = ["₹", "$", "€", "£"]


# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

def clean_column_names(df):
    """
    Standardize column names.
    """

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    # Handle duplicate column names
    new_columns = []
    counts = {}

    for column in df.columns:

        if column not in counts:
            counts[column] = 0
            new_columns.append(column)

        else:
            counts[column] += 1
            new_columns.append(
                f"{column}_{counts[column]}"
            )

    df.columns = new_columns

    return df


# ==========================================================
# STANDARDIZE TEXT VALUES
# ==========================================================

def standardize_text(df):
    """
    Remove unnecessary spaces and standardize text.

    Does not force capitalization changes because
    names and categories may have meaningful casing.
    """

    df = df.copy()

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            # Convert empty strings to missing values
            df[column] = df[column].replace(
                r"^\s*$",
                pd.NA,
                regex=True
            )

    return df


# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

def remove_empty_rows(df):
    """
    Remove rows where every column is missing.
    """

    return df.dropna(
        how="all"
    ).reset_index(drop=True)


# ==========================================================
# REMOVE EMPTY COLUMNS
# ==========================================================

def remove_empty_columns(df):
    """
    Remove columns where every value is missing.
    """

    return df.dropna(
        axis=1,
        how="all"
    )


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(df, subset=None):
    """
    Remove duplicate records.

    Default:
        Removes completely identical rows.

    Optional:
        Pass business key columns to identify
        duplicate records based on selected fields.
    """

    df = df.copy()

    if subset is not None:

        if isinstance(subset, str):
            subset = [subset]

        # Check that requested columns exist
        missing_columns = [
            col for col in subset
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Duplicate key columns not found: "
                f"{missing_columns}"
            )

    return df.drop_duplicates(
        subset=subset,
        keep="first"
    ).reset_index(drop=True)


# ==========================================================
# AUTOMATIC NUMERIC CONVERSION
# ==========================================================

def detect_numeric_columns(df, threshold=0.90):
    """
    Convert numeric-looking text columns into numeric types.

    Avoids converting ID, phone, postal and date columns.
    Handles commas and common currency symbols.
    """

    df = df.copy()

    for column in df.columns:

        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        column_name = column.lower()

        # Do not convert identifiers
        if any(
            keyword in column_name
            for keyword in ID_KEYWORDS
        ):
            continue

        # Do not process date columns here
        if any(
            keyword in column_name
            for keyword in DATE_KEYWORDS
        ):
            continue

        original = df[column]

        non_null_count = original.notna().sum()

        if non_null_count == 0:
            continue

        cleaned_values = (
            original.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace("%", "", regex=False)
        )

        converted = pd.to_numeric(
            cleaned_values,
            errors="coerce"
        )

        valid_count = converted.notna().sum()

        valid_percentage = (
            valid_count / non_null_count
        )

        if valid_percentage >= threshold:

            df[column] = converted

    return df


# ==========================================================
# AUTOMATIC DATETIME DETECTION
# ==========================================================

def detect_datetime_columns(df, threshold=0.80):
    """
    Automatically detect and convert date/time columns.

    Supports mixed date formats through Pandas.

    Date columns are detected using:
    - Column names
    - Percentage of valid date values
    - Existing datetime data types

    Invalid values are preserved as NaT after conversion.
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

        column_name = column.lower()

        is_date_column = any(
            keyword in column_name
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

            # Named date columns receive priority
            required_threshold = (
                threshold if is_date_column else 0.95
            )

            if valid_percentage >= required_threshold:

                df[column] = converted

        except (ValueError, TypeError, OverflowError):

            continue

    return df


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def fill_missing_values(
    df,
    fill_datetime=False,
    fill_numeric=True,
    fill_categorical=True
):
    """
    Handle missing values based on data type.

    Numeric:
        Median

    Datetime:
        Preserve missing values by default.
        Optional forward/backward fill.

    Categorical:
        Mode or Unknown.

    Completely empty columns should be removed
    before calling this function.
    """

    df = df.copy()

    for column in df.columns:

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):

            if fill_numeric:

                median = df[column].median()

                if pd.notna(median):

                    df[column] = df[column].fillna(
                        median
                    )

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

            if fill_categorical:

                mode = df[column].mode(
                    dropna=True
                )

                if not mode.empty:

                    df[column] = df[column].fillna(
                        mode.iloc[0]
                    )

                else:

                    df[column] = df[column].fillna(
                        "Unknown"
                    )

    return df


# ==========================================================
# DATA TYPE VALIDATION
# ==========================================================

def validate_data_types(df):
    """
    Return a summary of detected column data types.
    """

    result = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            dtype = "Datetime"

        elif pd.api.types.is_numeric_dtype(df[column]):
            dtype = "Numeric"

        else:
            dtype = "Categorical"

        result.append({
            "Column": column,
            "Data Type": dtype,
            "Pandas Type": str(df[column].dtype)
        })

    return pd.DataFrame(result)


# ==========================================================
# COMPLETE CLEANING PIPELINE
# ==========================================================

def clean_dataset(
    df,
    duplicate_subset=None,
    fill_datetime=False
):
    """
    Execute the complete data cleaning pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    duplicate_subset : list or str, optional
        Business key columns for duplicate detection.
        Defaults to exact duplicate row detection.

    fill_datetime : bool
        Whether to fill missing datetime values.
        Default is False to avoid inventing dates.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    cleaned = df.copy()

    # STEP 1: Clean column names
    cleaned = clean_column_names(cleaned)

    # STEP 2: Standardize text values
    cleaned = standardize_text(cleaned)

    # STEP 3: Remove completely empty rows
    cleaned = remove_empty_rows(cleaned)

    # STEP 4: Remove completely empty columns
    cleaned = remove_empty_columns(cleaned)

    # STEP 5: Remove duplicate records
    cleaned = remove_duplicates(
        cleaned,
        subset=duplicate_subset
    )

    # STEP 6: Detect and convert date columns
    cleaned = detect_datetime_columns(cleaned)

    # STEP 7: Convert numeric columns
    cleaned = detect_numeric_columns(cleaned)

    # STEP 8: Handle missing values
    cleaned = fill_missing_values(
        cleaned,
        fill_datetime=fill_datetime
    )

    # STEP 9: Reset index
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


# ==========================================================
# CLEANING REPORT
# ==========================================================

def cleaning_report(original_df, cleaned_df):
    """
    Generate a detailed before-and-after cleaning report.
    """

    # Normalize original column names for comparison
    original = clean_column_names(original_df.copy())

    # Basic row and column statistics
    report = {

        "Original Rows":
            int(len(original_df)),

        "Cleaned Rows":
            int(len(cleaned_df)),

        "Rows Removed":
            int(len(original_df) - len(cleaned_df)),

        "Original Columns":
            int(original_df.shape[1]),

        "Cleaned Columns":
            int(cleaned_df.shape[1]),

        "Columns Removed":
            int(original_df.shape[1] - cleaned_df.shape[1]),

        "Original Missing Values":
            int(original.isnull().sum().sum()),

        "Missing Values Remaining":
            int(cleaned_df.isnull().sum().sum()),

        "Missing Values Handled":
            int(
                original.isnull().sum().sum()
                - cleaned_df.isnull().sum().sum()
            ),

        "Original Duplicate Rows":
            int(original.duplicated().sum()),

        "Duplicate Rows Remaining":
            int(cleaned_df.duplicated().sum()),

        "Memory Usage (MB)":
            round(
                cleaned_df.memory_usage(deep=True).sum()
                / (1024 * 1024),
                2
            )
    }

    # Column type counts
    type_summary = validate_data_types(cleaned_df)

    report["Numeric Columns"] = int(
        (type_summary["Data Type"] == "Numeric").sum()
    )

    report["Categorical Columns"] = int(
        (type_summary["Data Type"] == "Categorical").sum()
    )

    report["Datetime Columns"] = int(
        (type_summary["Data Type"] == "Datetime").sum()
    )

    # List detected datetime columns
    report["Detected Date Columns"] = (
        type_summary.loc[
            type_summary["Data Type"] == "Datetime",
            "Column"
        ].tolist()
    )

    # List detected numeric columns
    report["Detected Numeric Columns"] = (
        type_summary.loc[
            type_summary["Data Type"] == "Numeric",
            "Column"
        ].tolist()
    )

    return report
