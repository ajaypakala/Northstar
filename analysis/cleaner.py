
"""
Northstar
Data Cleaning Module
"""

import pandas as pd


# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

def clean_column_names(df):
    """
    Clean and standardize column names.
    """

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.lower()
    )

    return df


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates().reset_index(drop=True)


# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

def remove_empty_rows(df):
    """
    Remove rows where every value is missing.
    """

    return df.dropna(how="all").reset_index(drop=True)


# ==========================================================
# REMOVE EMPTY COLUMNS
# ==========================================================

def remove_empty_columns(df):
    """
    Remove columns where every value is missing.
    """

    return df.dropna(axis=1, how="all")


# ==========================================================
# DETECT DATE COLUMNS
# ==========================================================

def detect_datetime_columns(df):
    """
    Automatically detect and convert date/time columns.

    Supports:
    - YYYY-MM-DD
    - DD-MM-YYYY
    - MM/DD/YYYY
    - Date and time values
    - Mixed date formats

    Columns containing date/time in their names receive
    priority during detection.
    """

    df = df.copy()

    for column in df.columns:

        # Skip columns that are already datetime
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue

        # Process only text-based columns
        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        original_values = df[column]

        # Ignore completely empty columns
        non_null_count = original_values.notna().sum()

        if non_null_count == 0:
            continue

        try:

            # Convert date values
            converted = pd.to_datetime(
                original_values,
                errors="coerce",
                format="mixed"
            )

            valid_dates = converted.notna().sum()

            # Calculate percentage of valid date values
            valid_percentage = (
                valid_dates / non_null_count
            ) * 100

            # Give priority to columns named date or time
            column_name = column.lower()

            is_date_column = (
                "date" in column_name
                or "time" in column_name
            )

            # Accept columns with at least 80% valid dates
            if valid_percentage >= 80:

                df[column] = converted

            # Date/time-named columns are also checked
            elif is_date_column:

                # Do not force conversion if values are
                # mostly invalid; preserve original data.

                continue

        except (ValueError, TypeError, OverflowError):

            continue

    return df


# ==========================================================
# FILL MISSING VALUES
# ==========================================================

def fill_missing_values(df):
    """
    Fill missing values automatically.

    Numeric columns  -> Median
    Datetime columns -> Forward fill and backward fill
    Text columns     -> Mode or Unknown
    """

    df = df.copy()

    for column in df.columns:

        # Numeric Columns
        if pd.api.types.is_numeric_dtype(df[column]):

            median = df[column].median()

            if pd.notna(median):

                df[column] = df[column].fillna(median)

        # Datetime Columns
        elif pd.api.types.is_datetime64_any_dtype(df[column]):

            df[column] = df[column].ffill().bfill()

        # Text Columns
        else:

            mode = df[column].mode()

            if not mode.empty:

                df[column] = df[column].fillna(
                    mode.iloc[0]
                )

            else:

                df[column] = df[column].fillna("Unknown")

    return df


# ==========================================================
# COMPLETE CLEANING PIPELINE
# ==========================================================

def clean_dataset(df):
    """
    Execute the complete data cleaning pipeline.
    """

    cleaned = df.copy()

    # Step 1: Standardize column names
    cleaned = clean_column_names(cleaned)

    # Step 2: Remove completely empty rows
    cleaned = remove_empty_rows(cleaned)

    # Step 3: Remove completely empty columns
    cleaned = remove_empty_columns(cleaned)

    # Step 4: Remove duplicate records
    cleaned = remove_duplicates(cleaned)

    # Step 5: Detect and convert date columns
    cleaned = detect_datetime_columns(cleaned)

    # Step 6: Fill missing values
    cleaned = fill_missing_values(cleaned)

    return cleaned


# ==========================================================
# CLEANING REPORT
# ==========================================================

def cleaning_report(original_df, cleaned_df):
    """
    Generate a report showing cleaning results.
    """

    report = {

        "Original Rows":
            len(original_df),

        "Cleaned Rows":
            len(cleaned_df),

        "Rows Removed":
            len(original_df) - len(cleaned_df),

        "Original Columns":
            original_df.shape[1],

        "Cleaned Columns":
            cleaned_df.shape[1],

        "Missing Values Remaining":
            int(cleaned_df.isnull().sum().sum()),

        "Duplicate Rows Remaining":
            int(cleaned_df.duplicated().sum()),

        "Memory Usage (MB)":
            round(
                cleaned_df.memory_usage(deep=True).sum()
                / (1024 * 1024),
                2
            )

    }

    return report
