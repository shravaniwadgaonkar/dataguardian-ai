import streamlit as st
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DataGuardian AI")
st.write("Intelligent Dataset Health, Bias & Privacy Auditor")

st.divider()


# =========================================================
# OPENAI
# =========================================================

client = None

if OpenAI is not None and "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
    except Exception:
        client = None


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload a CSV dataset to begin your analysis.")
    st.stop()


# =========================================================
# READ DATA
# =========================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

st.success("Dataset uploaded successfully!")


# =========================================================
# BASIC INFORMATION
# =========================================================

rows, columns = df.shape
duplicates = int(df.duplicated().sum())

missing_values = int(df.isnull().sum().sum())

numeric_columns = list(
    df.select_dtypes(include="number").columns
)


# =========================================================
# OUTLIER DETECTION
# =========================================================

total_outliers = 0
outlier_results = []

for column in numeric_columns:

    data = df[column].dropna()

    if len(data) == 0:
        continue

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        count = 0
        lower = q1
        upper = q3
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((data < lower) | (data > upper)).sum()
        )

    total_outliers += count

    outlier_results.append({
        "Column": column,
        "Outliers": count,
        "Lower Bound": round(lower, 2),
        "Upper Bound": round(upper, 2)
    })


# =========================================================
# PRIVACY AUDIT
# =========================================================

sensitive_keywords = [
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "dob",
    "birth",
    "aadhaar",
    "aadhar",
    "pan",
    "passport",
    "pincode",
    "zip",
    "postal"
]

sensitive_columns = []

for column in df.columns:

    name = str(column).lower()

    if any(
        keyword in name
        for keyword in sensitive_keywords
    ):
        sensitive_columns.append(column)


if len(sensitive_columns) >= 3:
    privacy_risk = "HIGH"
elif len(sensitive_columns) > 0:
    privacy_risk = "MEDIUM"
else:
    privacy_risk = "LOW"


# =========================================================
# HEALTH SCORE
# =========================================================

score = 100.0

# Missing data penalty
if rows > 0 and columns > 0:

    missing_percentage = (
        missing_values / (rows * columns)
    ) * 100

    score -= min(
        missing_percentage * 2,
        30
    )


# Duplicate penalty
if rows > 0:

    duplicate_percentage = (
        duplicates / rows
    ) * 100

    score -= min(
        duplicate_percentage * 2,
        20
    )


# Outlier penalty
if len(numeric_columns) > 0:

    possible_values = (
        rows * len(numeric_columns)
    )

    if possible_values > 0:

        outlier_percentage = (
            total_outliers / possible_values
        ) * 100

        score -= min(
            outlier_percentage * 2,
            40
        )


# Privacy penalty
if privacy_risk == "HIGH":
    score -= 15
elif privacy_risk == "MEDIUM":
    score -= 7


score = max(
    0,
    min(100, score)
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Rows", rows)

with c2:
    st.metric("Columns", columns)

with c3:
    st.metric(
        "Duplicate Rows",
        duplicates
    )


# =========================================================
# DATA QUALITY
# =========================================================

st.subheader("🔍 Data Quality")

if missing_values == 0:
    st.success("✅ No missing values detected.")
else:
    st.warning(
        f"⚠️ {missing_values} missing values detected."
    )

if duplicates == 0:
    st.success("✅ No duplicate rows detected.")
else:
    st.warning(
        f"⚠️ {duplicates} duplicate rows detected."
    )


# =========================================================
# COLUMN INFORMATION
# =========================================================

st.subheader("🧩 Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": [
        df[column].nunique()
        for column in df.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True
)


# =========================================================
# OUTLIER DETECTION
# =========================================================

st.subheader("🚨 Outlier Detection")

if outlier_results:

    outlier_table = pd.DataFrame(
        outlier_results
    )

    st.dataframe(
        outlier_table,
        use_container_width=True
    )

    if total_outliers > 0:

        st.warning(
            f"⚠️ {total_outliers} "
            "potential outlier values detected."
        )

    else:

        st.success(
            "✅ No potential outliers detected."
        )

else:

    st.info(
        "No numerical columns found."
    )


# =========================================================
# PRIVACY
# =========================================================

st.subheader("🔐 Privacy Risk Audit")

if sensitive_columns:

    st.warning(
        "⚠️ Potentially sensitive columns detected."
    )

    for column in sensitive_columns:
        st.write(f"🔒 `{column}`")

else:

    st.success(
        "✅ No potentially sensitive "
        "column names detected."
    )

st.caption(
    "Privacy detection is based on column names "
    "and is a heuristic, not a guarantee."
)


# =========================================================
# BIAS AUDIT
# =========================================================

st.subheader("⚖️ Bias Audit")

categorical_columns = list(
    df.select_dtypes(
        include=["object", "category"]
    ).columns
)

bias_available = False
bias_status = "⚪ Not Evaluated"

if categorical_columns:

    group_column = st.selectbox(
        "Select group column",
        categorical_columns,
        key="group_column"
    )

    target_column = st.selectbox(
        "Select outcome column",
        list(df.columns),
        key="target_column"
    )

    unique_values = (
        df[target_column]
        .dropna()
        .unique()
    )

    if len(unique_values) == 2:

        bias_available = True
        bias_status = "🟢 Evaluated"

        positive_value = st.selectbox(
            "Select positive outcome",
            unique_values,
            key="positive_value"
        )

        temp = df[
            [group_column, target_column]
        ].dropna()

        temp["positive"] = (
            temp[target_column] == positive_value
        )

        bias_table = (
            temp
            .groupby(group_column)["positive"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )

        bias_table.columns = [
            group_column,
            "Outcome Rate (%)"
        ]

        st.dataframe(
            bias_table,
            use_container_width=True
        )

        if len(bias_table) >= 2:

            difference = (
                bias_table["Outcome Rate (%)"].max()
                -
                bias_table["Outcome Rate (%)"].min()
            )

            st.metric(
                "Maximum Outcome Rate Difference",
                f"{difference:.2f}%"
            )

            if difference >= 20:
                st.error(
                    "🔴 Large outcome disparity detected."
                )
            elif difference >= 10:
                st.warning(
                    "🟡 Potential outcome disparity detected."
                )
            else:
                st.success(
                    "🟢 No large outcome disparity detected."
                )

        st.caption(
            "A disparity does not automatically prove discrimination."
        )

    else:

        st.info(
            "Bias analysis currently supports "
            "binary outcomes."
        )

else:

    st.info(
        "No categorical columns found. "
        "Bias analysis cannot be performed automatically."
    )


# =========================================================
# RISK REPORT
# =========================================================

st.subheader("🛡️ Dataset Risk Report")

if score >= 85:
    quality_status = "🟢 Excellent"
elif score >= 70:
    quality_status = "🟡 Good"
elif score >= 50:
    quality_status = "🟠 Needs Attention"
else:
    quality_status = "🔴 Poor"


if total_outliers == 0:
    outlier_status = "🟢 Low"
elif total_outliers < rows * 0.10:
    outlier_status = "🟡 Moderate"
else:
    outlier_status = "🔴 High"


st.write(
    f"**Overall Health:** {score:.1f}/100"
)

st.write(
    f"**Data Quality:** {quality_status}"
)

st.write(
    f"**Outlier Risk:** {outlier_status}"
)

st.write(
    f"**Privacy Risk:** {privacy_risk}"
)

st.write(
    f"**Bias Status:** {bias_status}"
)


# =========================================================
# DATA STATISTICS
# =========================================================

st.subheader("📈 Dataset Statistics")

if numeric_columns:

    selected_column = st.selectbox(
        "Select numerical column",
        numeric_columns,
        key="statistics_column"
    )

    chart_data = df[
        [selected_column]
    ].dropna()

    st.bar_chart(
        chart_data,
        x=None,
        y=selected_column
    )

else:

    st.info(
        "No numerical columns available for statistics."
    )


# =========================================================
# AI DATA ANALYST
# =========================================================

st.subheader("🤖 AI Data Analyst")

st.write(
    "Generate an automated analysis of your dataset."
)

if st.button(
    "🤖 Generate AI Analysis",
    type="primary"
):

    ai_report = None

    summary = f"""
Dataset: {uploaded_file.name}
Rows: {rows}
Columns: {columns}
Missing values: {missing_values}
Duplicate rows: {duplicates}
Potential outliers: {total_outliers}
Sensitive columns: {sensitive_columns}
Privacy risk: {privacy_risk}
Health score: {score:.1f}/100
Bias status: {bias_status}
"""


    # =====================================================
    # REAL OPENAI ANALYSIS
    # =====================================================

    if client is not None:

        try:

            with st.spinner(
                "🤖 AI is analyzing your dataset..."
            ):

                response = client.responses.create(

                    model="gpt-5-mini",

                    input=f"""
You are a professional Data Scientist.

Analyze the following dataset summary:

{summary}

Create a concise report with:

1. Dataset Overview
2. Data Quality
3. Outlier Analysis
4. Privacy
5. Bias
6. Machine Learning Readiness
7. Recommendations

Do not invent information.
"""
                )

            ai_report = response.output_text

        except Exception:
            ai_report = None


    # =====================================================
    # LOCAL FALLBACK
    # =====================================================

    if ai_report is None:

        st.info(
            "ℹ️ OpenAI is unavailable or has no "
            "remaining credits. DataGuardian's "
            "local analysis engine is being used."
        )

        report = []

        report.append(
            "## 📋 Dataset Overview"
        )

        report.append(
            f"The dataset contains **{rows} rows** "
            f"and **{columns} columns**."
        )


        report.append(
            "## 🔍 Data Quality"
        )

        if missing_values == 0:

            report.append(
                "✅ No missing values detected."
            )

        else:

            report.append(
                f"⚠️ {missing_values} missing "
                "values require attention."
            )


        if duplicates == 0:

            report.append(
                "✅ No duplicate records detected."
            )

        else:

            report.append(
                f"⚠️ {duplicates} duplicate "
                "records detected."
            )


        report.append(
            "## 🚨 Outlier Analysis"
        )

        if total_outliers > 0:

            report.append(
                f"⚠️ **{total_outliers} potential "
                "outlier values** were detected."
            )

            report.append(
                "These values should be investigated "
                "before machine-learning training."
            )

        else:

            report.append(
                "✅ No potential outliers detected."
            )


        report.append(
            "## 🔐 Privacy"
        )

        if sensitive_columns:

            report.append(
                "Potentially sensitive columns: "
                +
                ", ".join(
                    map(str, sensitive_columns)
                )
            )

        else:

            report.append(
                "✅ No potentially sensitive "
                "column names detected."
            )


        report.append(
            "## ⚖️ Bias"
        )

        if bias_available:

            report.append(
                "Bias analysis was performed "
                "using the selected group and outcome."
            )

        else:

            report.append(
                "Bias analysis was not performed "
                "because suitable categorical data "
                "was unavailable."
            )


        report.append(
            "## 🤖 Machine Learning Readiness"
        )

        if score >= 85:

            report.append(
                "The dataset is in good condition "
                "for initial machine-learning experiments."
            )

        elif score >= 70:

            report.append(
                "The dataset is usable, but "
                "some preprocessing is recommended."
            )

        else:

            report.append(
                "Additional preprocessing is "
                "recommended before ML training."
            )


        report.append(
            "## 💡 Recommendations"
        )

        if total_outliers > 0:
            report.append(
                "• Investigate potential outliers."
            )

        if missing_values > 0:
            report.append(
                "• Handle missing values."
            )

        if duplicates > 0:
            report.append(
                "• Review duplicate records."
            )

        if sensitive_columns:
            report.append(
                "• Consider anonymizing sensitive data."
            )

        report.append(
            "• Perform exploratory data analysis "
            "before model training."
        )

        ai_report = "\n\n".join(report)


    # =====================================================
    # SHOW REPORT
    # =====================================================

    st.markdown(ai_report)


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    full_report = f"""
DATAGUARDIAN AI
DATASET ANALYSIS REPORT

Dataset: {uploaded_file.name}

Rows: {rows}
Columns: {columns}

Missing Values: {missing_values}
Duplicate Rows: {duplicates}
Potential Outliers: {total_outliers}

Privacy Risk: {privacy_risk}

Overall Health: {score:.1f}/100

Bias Status: {bias_status}

--------------------------------

{ai_report}
"""


    st.download_button(
        "📥 Download Analysis Report",
        full_report,
        file_name="DataGuardian_Report.txt",
        mime="text/plain"
    )


# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# FINAL SCORE
# =========================================================

st.subheader("🏥 Dataset Health Score")

st.metric(
    "Overall Dataset Health",
    f"{score:.1f}/100"
)

if score >= 85:

    st.success(
        "🟢 Dataset health is excellent."
    )

elif score >= 70:

    st.warning(
        "🟡 Dataset is usable but needs some attention."
    )

elif score >= 50:

    st.warning(
        "🟠 Dataset requires cleaning before ML use."
    )

else:

    st.error(
        "🔴 Dataset requires significant preprocessing."
    )
