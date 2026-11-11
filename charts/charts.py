
"""
Northstar
Interactive Charts Module
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# COMMON CHART SETTINGS
# ==========================================================

CHART_TEMPLATE = "plotly_white"


def improve_layout(fig):
    """
    Apply common styling to charts.
    """

    fig.update_layout(
        template=CHART_TEMPLATE,
        title_x=0.02,
        title_font_size=20,
        font=dict(
            family="Arial",
            size=12
        ),
        margin=dict(
            l=50,
            r=30,
            t=70,
            b=50
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13
        )
    )

    return fig


# ==========================================================
# HISTOGRAM
# ==========================================================

def histogram_chart(df, column):

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
        template=CHART_TEMPLATE,
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        bargap=0.08
    )

    return improve_layout(fig)


# ==========================================================
# BOX PLOT
# ==========================================================

def box_plot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Box Plot - {column}",
        template=CHART_TEMPLATE,
        points="outliers"
    )

    fig.update_layout(
        yaxis_title=column
    )

    return improve_layout(fig)


# ==========================================================
# BAR CHART
# ==========================================================

def bar_chart(df, column):

    data = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    data.columns = [column, "Count"]

    # Limit excessive categories for readability
    data = data.head(20)

    fig = px.bar(
        data,
        x=column,
        y="Count",
        text="Count",
        title=f"{column} Distribution",
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Number of Records"
    )

    return improve_layout(fig)


# ==========================================================
# PIE CHART
# ==========================================================

def pie_chart(df, column):

    data = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    data.columns = [column, "Count"]

    # Avoid overcrowding
    data = data.head(10)

    fig = px.pie(
        data,
        names=column,
        values="Count",
        hole=0.45,
        title=f"{column} Distribution",
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "%{label}<br>"
            "Count: %{value}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )
    )

    return improve_layout(fig)


# ==========================================================
# SCATTER PLOT
# ==========================================================

def scatter_chart(df, x_column, y_column):
    """
    Display relationships between two numeric columns.
    """

    temp = df[[x_column, y_column]].copy()

    temp[x_column] = pd.to_numeric(
        temp[x_column],
        errors="coerce"
    )

    temp[y_column] = pd.to_numeric(
        temp[y_column],
        errors="coerce"
    )

    temp = temp.dropna()

    fig = px.scatter(
        temp,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
        template=CHART_TEMPLATE,
        opacity=0.60,
        render_mode="webgl"
    )

    fig.update_traces(
        marker=dict(
            size=8
        ),
        hovertemplate=(
            f"{x_column}: %{{x}}<br>"
            f"{y_column}: %{{y}}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    return improve_layout(fig)


# ==========================================================
# LINE CHART
# ==========================================================

def line_chart(df, x_column, y_column):
    """
    Create a monthly aggregated time-series chart.

    Numeric values are summed monthly to avoid plotting
    hundreds of individual order records.
    """

    temp = df[[x_column, y_column]].copy()

    # Convert date column
    temp[x_column] = pd.to_datetime(
        temp[x_column],
        errors="coerce"
    )

    # Convert numeric column
    temp[y_column] = pd.to_numeric(
        temp[y_column],
        errors="coerce"
    )

    # Remove invalid records
    temp = temp.dropna(
        subset=[x_column, y_column]
    )

    if temp.empty:
        return improve_layout(
            go.Figure().update_layout(
                title="No valid data available for time series"
            )
        )

    # Group records by month
    temp["Month"] = (
        temp[x_column]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_data = (
        temp.groupby("Month", as_index=False)[y_column]
        .sum()
        .sort_values("Month")
    )

    fig = px.line(
        monthly_data,
        x="Month",
        y=y_column,
        markers=True,
        title=f"Monthly {y_column} Trend",
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "Month: %{x|%b %Y}<br>"
            f"{y_column}: %{{y:,.2f}}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=y_column,
        hovermode="x unified"
    )

    fig.update_xaxes(
        tickformat="%b %Y",
        tickangle=-30
    )

    return improve_layout(fig)


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(correlation_df):

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns,
            y=correlation_df.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            text=correlation_df.round(2).values,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "Correlation: %{z:.2f}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Correlation"
            )
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features"
    )

    return improve_layout(fig)


# ==========================================================
# MISSING VALUE CHART
# ==========================================================

def missing_values_chart(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:
        return None

    chart = missing.reset_index()

    chart.columns = [
        "Column",
        "Missing Values"
    ]

    fig = px.bar(
        chart,
        x="Column",
        y="Missing Values",
        text="Missing Values",
        title="Missing Values by Column",
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    return improve_layout(fig)


# ==========================================================
# DATA TYPE CHART
# ==========================================================

def data_type_chart(df):

    data = (
        df.dtypes
        .astype(str)
        .value_counts()
        .reset_index()
    )

    data.columns = [
        "Data Type",
        "Count"
    ]

    fig = px.pie(
        data,
        names="Data Type",
        values="Count",
        hole=0.5,
        title="Column Data Types",
        template=CHART_TEMPLATE
    )

    fig.update_traces(
        textinfo="percent+label"
    )

    return improve_layout(fig)


# ==========================================================
# CORRELATION SCATTER
# ==========================================================

def correlation_scatter(df):

    numeric = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric) < 2:
        return None

    fig = px.scatter(
        df,
        x=numeric[0],
        y=numeric[1],
        title=f"{numeric[1]} vs {numeric[0]}",
        template=CHART_TEMPLATE,
        opacity=0.6
    )

    return improve_layout(fig)


# ==========================================================
# TIME SERIES CHART
# ==========================================================

def time_series_chart(df, date_column, value_column):
    """
    Reuse the monthly aggregation logic from line_chart.
    """

    return line_chart(
        df,
        date_column,
        value_column
    )
