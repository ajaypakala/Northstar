
"""
Northstar
Utility Functions
"""

from datetime import datetime
import pandas as pd


# ==========================================================
# FILE INFORMATION
# ==========================================================

def get_file_name(uploaded_file):
    """
    Returns uploaded file name.
    """
    return uploaded_file.name


def get_file_size(uploaded_file):
    """
    Returns file size in MB.
    """
    return round(
        uploaded_file.size / (1024 * 1024), 2
    )


def get_file_extension(uploaded_file):
    """
    Returns file extension.
    """
    return uploaded_file.name.split(".")[-1].lower()


# ==========================================================
# COLUMN HELPERS
# ==========================================================

def get_numeric_columns(df):
    """
    Identify numeric columns.
    Excludes boolean columns.
    """

    return [
        column
        for column in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        )
    ]


def get_categorical_columns(df):
    """
    Identify categorical and text columns.

    Supports:
    - object
    - string
    - category

    Excludes numeric and datetime columns.
    """

    categorical_columns = []

    for column in df.columns:

        dtype = df[column].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):

            categorical_columns.append(column)

    return categorical_columns


def get_datetime_columns(df):
    """
    Identify datetime columns.

    Supports standard and timezone-aware datetime types.
    """

    datetime_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            datetime_columns.append(column)

    return datetime_columns


# ==========================================================
# DATA PREVIEW
# ==========================================================

def preview_dataframe(df, rows=10):
    """
    Return selected number of rows.
    """

    return df.head(rows)


# ==========================================================
# DATASET SUMMARY FOR AI
# ==========================================================

def generate_dataset_summary(df):
    """
    Create dataset summary for Gemini.
    """

    summary = []

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    datetime_columns = get_datetime_columns(df)

    summary.append("DATASET INFORMATION\n")

    summary.append(f"Rows : {len(df)}")

    summary.append(f"Columns : {len(df.columns)}")

    summary.append(
        f"Missing Values : {int(df.isnull().sum().sum())}"
    )

    summary.append(
        f"Duplicate Rows : {int(df.duplicated().sum())}"
    )

    summary.append("\nCOLUMN NAMES")

    summary.append(", ".join(df.columns))

    summary.append("\nNUMERIC COLUMNS")

    summary.append(", ".join(numeric_columns))

    summary.append("\nCATEGORICAL COLUMNS")

    summary.append(", ".join(categorical_columns))

    summary.append("\nDATETIME COLUMNS")

    summary.append(", ".join(datetime_columns))

    summary.append("\nDATA TYPES")

    summary.append(df.dtypes.to_string())

    summary.append("\nSUMMARY STATISTICS")

    try:

        summary.append(
            df.describe(include="all").to_string()
        )

    except Exception:

        pass

    summary.append("\nFIRST FIVE ROWS")

    summary.append(
        df.head().to_string()
    )

    return "\n".join(summary)


# ==========================================================
# FORMAT NUMBERS
# ==========================================================

def format_number(value):
    """
    Format large numbers into readable values.
    """

    try:

        value = float(value)

    except (ValueError, TypeError):

        return value

    if value >= 1_000_000_000:

        return f"{value / 1_000_000_000:.2f}B"

    elif value >= 1_000_000:

        return f"{value / 1_000_000:.2f}M"

    elif value >= 1_000:

        return f"{value / 1_000:.2f}K"

    else:

        return f"{value:.2f}"


# ==========================================================
# EXPORT CSV
# ==========================================================

def dataframe_to_csv(df):
    """
    Convert DataFrame into downloadable CSV.
    """

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ==========================================================
# REPORT NAME
# ==========================================================

def generate_report_filename():
    """
    Generate unique PDF report filename.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return f"Northstar_Report_{timestamp}.pdf"


# ==========================================================
# MEMORY USAGE
# ==========================================================

def memory_usage(df):
    """
    Calculate DataFrame memory usage in MB.
    """

    return round(
        df.memory_usage(deep=True).sum()
        / (1024 * 1024),
        2
    )


# ==========================================================
# MISSING PERCENTAGE
# ==========================================================

def missing_percentage(df):
    """
    Calculate missing percentage for every column.
    """

    if len(df) == 0:

        return pd.Series(
            0.0,
            index=df.columns
        )

    return (
        df.isnull().sum()
        / len(df)
        * 100
    ).round(2)


# ==========================================================
# UNIQUE VALUES
# ==========================================================

def unique_value_summary(df):
    """
    Calculate unique values for each column.
    """

    result = pd.DataFrame({

        "Column": df.columns,

        "Unique Values": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ]

    })

    return result


# ==========================================================
# DATA TYPE SUMMARY
# ==========================================================

def datatype_summary(df):
    """
    Generate column data type summary.
    """

    result = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            (
                "Datetime"
                if pd.api.types.is_datetime64_any_dtype(df[col])
                else "Numeric"
                if pd.api.types.is_numeric_dtype(df[col])
                else "Categorical"
            )
            for col in df.columns
        ],

        "Pandas Type": df.dtypes.astype(str).values

    })

    return result
