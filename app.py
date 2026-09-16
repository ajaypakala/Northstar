"""
InsightGPT
Enterprise AI Data Analyst
"""

import streamlit as st

from config import (
    APP_NAME,
    APP_ICON,
    APP_LAYOUT,
    DEFAULT_PREVIEW_ROWS
)

from analysis.analyzer import (
    load_data,
    dataset_profile,
    numerical_summary,
    categorical_summary,
    correlation_matrix
)

from analysis.cleaner import (
    clean_dataset,
    cleaning_report
)

from utils.helpers import (
    get_numeric_columns,
    get_categorical_columns,
    get_datetime_columns,
    preview_dataframe,
    generate_dataset_summary,
    dataframe_to_csv,
    generate_report_filename
)

from ai.ai_service import (
    generate_ai_insights,
    chat_with_data
)

from reports.pdf_report import (
    create_pdf_report
)

from charts.charts import (
    histogram_chart,
    box_plot,
    bar_chart,
    pie_chart,
    scatter_chart,
    line_chart,
    correlation_heatmap,
    missing_values_chart,
    data_type_chart
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=APP_LAYOUT
)

st.title(APP_NAME)

st.caption(
    "Enterprise AI Powered Data Analysis Platform"
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload a CSV file to begin.")

    st.stop()

# ==========================================================
# LOAD DATA
# ==========================================================

try:

    raw_df = load_data(uploaded_file)

except Exception as e:

    st.error(e)

    st.stop()

# ==========================================================
# CLEAN DATA
# ==========================================================

clean_df = clean_dataset(raw_df)

report = cleaning_report(
    raw_df,
    clean_df
)

df = clean_df.copy()

# ==========================================================
# PROFILE
# ==========================================================

profile = dataset_profile(df)

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Rows",
    profile["rows"]
)

col2.metric(
    "Columns",
    profile["columns"]
)

col3.metric(
    "Missing",
    profile["missing"]
)

col4.metric(
    "Duplicates",
    profile["duplicates"]
)

col5.metric(
    "Memory",
    f"{profile['memory_mb']} MB"
)

st.divider()

# ==========================================================
# CLEANING REPORT
# ==========================================================

st.header("🧹 Cleaning Report")

st.json(report)

st.divider()

# ==========================================================
# DATA PREVIEW
# ==========================================================

st.header("👀 Dataset Preview")

preview_rows = st.slider(

    "Rows",

    5,

    50,

    DEFAULT_PREVIEW_ROWS

)

st.dataframe(

    preview_dataframe(

        df,

        preview_rows

    ),

    use_container_width=True

)

st.divider()

# ==========================================================
# COLUMN TYPES
# ==========================================================

numeric_columns = get_numeric_columns(df)

categorical_columns = get_categorical_columns(df)

datetime_columns = get_datetime_columns(df)

left, right = st.columns(2)

with left:

    st.subheader("Numeric Columns")

    st.write(numeric_columns)

    st.subheader("Datetime Columns")

    st.write(datetime_columns)

with right:

    st.subheader("Categorical Columns")

    st.write(categorical_columns)

st.divider()

# ==========================================================
# NUMERICAL SUMMARY
# ==========================================================

if numeric_columns:

    st.header("📈 Numerical Summary")

    st.dataframe(

        numerical_summary(df),

        use_container_width=True

    )

# ==========================================================
# CATEGORICAL SUMMARY
# ==========================================================

if categorical_columns:

    st.header("📋 Categorical Summary")

    selected_column = st.selectbox(

        "Select Column",

        categorical_columns

    )

    st.dataframe(

        categorical_summary(

            df,

            selected_column

        ),

        use_container_width=True

    )

st.divider()


# ==========================================================
# VISUALIZATIONS
# ==========================================================

st.header("📊 Interactive Visualizations")

# ----------------------------------------------------------
# HISTOGRAM
# ----------------------------------------------------------

if numeric_columns:

    st.subheader("Histogram")

    histogram_column = st.selectbox(
        "Select Numeric Column",
        numeric_columns,
        key="histogram"
    )

    st.plotly_chart(
        histogram_chart(
            df,
            histogram_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# BOX PLOT
# ----------------------------------------------------------

if numeric_columns:

    st.subheader("Box Plot")

    box_column = st.selectbox(
        "Select Numeric Column",
        numeric_columns,
        key="boxplot"
    )

    st.plotly_chart(
        box_plot(
            df,
            box_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# BAR CHART
# ----------------------------------------------------------

if categorical_columns:

    st.subheader("Bar Chart")

    bar_column = st.selectbox(
        "Select Categorical Column",
        categorical_columns,
        key="bar"
    )

    st.plotly_chart(
        bar_chart(
            df,
            bar_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# PIE CHART
# ----------------------------------------------------------

if categorical_columns:

    st.subheader("Pie Chart")

    pie_column = st.selectbox(
        "Select Categorical Column",
        categorical_columns,
        key="pie"
    )

    st.plotly_chart(
        pie_chart(
            df,
            pie_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# SCATTER PLOT
# ----------------------------------------------------------

if len(numeric_columns) >= 2:

    st.subheader("Scatter Plot")

    col1, col2 = st.columns(2)

    with col1:

        x_column = st.selectbox(
            "X Axis",
            numeric_columns,
            key="scatter_x"
        )

    with col2:

        y_column = st.selectbox(
            "Y Axis",
            numeric_columns,
            index=1,
            key="scatter_y"
        )

    st.plotly_chart(
        scatter_chart(
            df,
            x_column,
            y_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# LINE CHART
# ----------------------------------------------------------

if datetime_columns and numeric_columns:

    st.subheader("Time Series")

    date_column = st.selectbox(
        "Date Column",
        datetime_columns,
        key="date_column"
    )

    value_column = st.selectbox(
        "Numeric Column",
        numeric_columns,
        key="value_column"
    )

    st.plotly_chart(
        line_chart(
            df,
            date_column,
            value_column
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------------

if len(numeric_columns) >= 2:

    st.subheader("Correlation Heatmap")

    corr = correlation_matrix(df)

    if not corr.empty:

        st.plotly_chart(
            correlation_heatmap(corr),
            use_container_width=True
        )

# ----------------------------------------------------------
# MISSING VALUES
# ----------------------------------------------------------

missing_chart = missing_values_chart(df)

if missing_chart is not None:

    st.subheader("Missing Values")

    st.plotly_chart(
        missing_chart,
        use_container_width=True
    )

# ----------------------------------------------------------
# DATA TYPES
# ----------------------------------------------------------

st.subheader("Column Data Types")

st.plotly_chart(
    data_type_chart(df),
    use_container_width=True
)

st.divider()

# ==========================================================
# AI DATA ANALYSIS
# ==========================================================

st.header("🤖 AI Data Analyst")

dataset_summary = generate_dataset_summary(df)

st.info(
    "Generate AI-powered insights for your uploaded dataset."
)

# ----------------------------------------------------------
# GENERATE AI REPORT
# ----------------------------------------------------------

if st.button(
    "Generate AI Report",
    use_container_width=True
):

    with st.spinner("Analyzing dataset..."):

        ai_report = generate_ai_insights(
            dataset_summary
        )

    st.session_state["ai_report"] = ai_report

    st.success("AI Report Generated Successfully")

    st.markdown(ai_report)

# ----------------------------------------------------------
# SHOW PREVIOUS REPORT
# ----------------------------------------------------------

elif "ai_report" in st.session_state:

    st.markdown(
        st.session_state["ai_report"]
    )

st.divider()

# ==========================================================
# CHAT WITH DATASET
# ==========================================================

st.header("💬 Chat With Your Dataset")

question = st.text_area(

    "Ask a question about your dataset",

    placeholder="""
Examples

• Summarize this dataset

• What are the main insights?

• Which columns have missing values?

• Suggest data cleaning steps.

• Recommend business improvements.
""",

    height=150

)

if st.button(
    "Ask AI",
    use_container_width=True
):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Thinking..."
        ):

            answer = chat_with_data(
                dataset_summary,
                question
            )

        st.success("Answer")

        st.markdown(answer)

st.divider()

# ==========================================================
# EXPORT SECTION
# ==========================================================

st.header("📥 Export")

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# DOWNLOAD CLEANED CSV
# ----------------------------------------------------------

with col1:

    st.download_button(

        label="⬇ Download Cleaned CSV",

        data=dataframe_to_csv(df),

        file_name="cleaned_dataset.csv",

        mime="text/csv",

        use_container_width=True

    )

# ----------------------------------------------------------
# DOWNLOAD PDF REPORT
# ----------------------------------------------------------

with col2:

    if "ai_report" in st.session_state:

        pdf_name = generate_report_filename()

        create_pdf_report(

            pdf_name,

            profile,

            st.session_state["ai_report"]

        )

        with open(pdf_name, "rb") as pdf:

            st.download_button(

                label="📄 Download PDF Report",

                data=pdf,

                file_name=pdf_name,

                mime="application/pdf",

                use_container_width=True

            )

    else:

        st.info(
            "Generate the AI Report first."
        )

st.divider()


# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.header("📋 Dataset Information")

tab1, tab2, tab3 = st.tabs([
    "Columns",
    "Data Types",
    "Missing Values"
])

# ----------------------------------------------------------
# COLUMN LIST
# ----------------------------------------------------------

with tab1:

    st.dataframe(
        df.columns.to_frame(
            name="Column Name"
        ),
        use_container_width=True
    )

# ----------------------------------------------------------
# DATA TYPES
# ----------------------------------------------------------

with tab2:

    dtype_df = df.dtypes.astype(str).reset_index()

    dtype_df.columns = [
        "Column",
        "Data Type"
    ]

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

# ----------------------------------------------------------
# MISSING VALUES
# ----------------------------------------------------------

with tab3:

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_df,
        use_container_width=True
    )

st.divider()

# ==========================================================
# QUICK DATA PREVIEW
# ==========================================================

st.header("🔍 First 20 Rows")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.divider()

# ==========================================================
# APPLICATION INFORMATION
# ==========================================================

with st.expander("ℹ About Northstar"):

    st.markdown("""
### Northstar – Enterprise AI Data Analyst

Features

- Upload any CSV dataset
- Automatic data cleaning
- Dataset profiling
- Interactive visualizations
- AI-powered insights using Gemini
- Chat with your dataset
- Export cleaned CSV
- Generate PDF report

Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- Google Gemini
- ReportLab
""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
---
<div style="text-align:center">

### 🚀 Northstar - Enterprise AI Data Analyst

Built with ❤️ using

**Python • Streamlit • Pandas • Plotly • Google Gemini**

Version 1.0

</div>
""",
    unsafe_allow_html=True
)