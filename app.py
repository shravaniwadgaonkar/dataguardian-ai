import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =========================================================
# PAGE CONFIG
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

missing_values = int(
    df.isnull().sum().sum()
)

numeric_columns = list(
    df.select_dtypes(include="number").columns
)

categorical_columns = list(
    df.select_dtypes(
        include=["object", "category"]
    ).columns
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
            (
                (data < lower) |
                (data > upper)
            ).sum()
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

    column_name = str(column).lower()

    if any(
        keyword in column_name
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
# DATA QUALITY CALCULATIONS
# =========================================================

if rows > 0 and columns > 0:

    missing_percentage = (
        missing_values /
        (rows * columns)
    ) * 100

else:

    missing_percentage = 0


if rows > 0:

    duplicate_percentage = (
        duplicates / rows
    ) * 100

else:

    duplicate_percentage = 0


# =========================================================
# OUTLIER BURDEN
# =========================================================

if rows > 0:

    outlier_burden = (
        total_outliers / rows
    ) * 100

else:

    outlier_burden = 0


# =========================================================
# HEALTH SCORE
# =========================================================

score = 100.0

missing_penalty = min(
    missing_percentage * 2,
    25
)

score -= missing_penalty


duplicate_penalty = min(
    duplicate_percentage * 2,
    15
)

score -= duplicate_penalty


outlier_penalty = min(
    outlier_burden * 0.8,
    40
)

score -= outlier_penalty


if privacy_risk == "HIGH":

    score -= 15

elif privacy_risk == "MEDIUM":

    score -= 7


score = max(
    0,
    min(100, score)
)


# =========================================================
# BIAS AUDIT
# =========================================================

bias_available = False
bias_status = "⚪ Not Evaluated"
bias_difference = None
bias_table = None

st.subheader("⚖️ Bias Audit")

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
        ].dropna().copy()

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

            bias_difference = (
                bias_table["Outcome Rate (%)"].max()
                -
                bias_table["Outcome Rate (%)"].min()
            )

            st.metric(
                "Maximum Outcome Rate Difference",
                f"{bias_difference:.2f}%"
            )

            if bias_difference >= 20:

                st.error(
                    "🔴 Large outcome disparity detected."
                )

            elif bias_difference >= 10:

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
            "Bias analysis currently supports binary outcomes."
        )

else:

    st.info(
        "No categorical columns found. "
        "Bias analysis cannot be performed automatically."
    )


# =========================================================
# RISK LEVELS
# =========================================================

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

elif outlier_burden < 5:

    outlier_status = "🟡 Moderate"

else:

    outlier_status = "🔴 High"


if privacy_risk == "LOW":

    privacy_display = "🟢 Low"

elif privacy_risk == "MEDIUM":

    privacy_display = "🟡 Medium"

else:

    privacy_display = "🔴 High"


# =========================================================
# ML READINESS
# =========================================================

if score >= 85:

    ml_readiness = "🟢 Ready for initial ML experiments"

elif score >= 70:

    ml_readiness = "🟡 Usable after preprocessing"

elif score >= 50:

    ml_readiness = "🟠 Needs significant preprocessing"

else:

    ml_readiness = "🔴 Not recommended before cleaning"


# =========================================================
# DATASET RISK REPORT
# =========================================================

st.subheader("🛡️ Dataset Risk Report")

r1, r2, r3, r4 = st.columns(4)

with r1:

    st.metric(
        "Overall Health",
        f"{score:.1f}/100"
    )

with r2:

    st.metric(
        "Data Quality",
        quality_status
    )

with r3:

    st.metric(
        "Outlier Risk",
        outlier_status
    )

with r4:

    st.metric(
        "Privacy Risk",
        privacy_display
    )


r5, r6 = st.columns(2)

with r5:

    st.write(
        f"**⚖️ Bias Status:** {bias_status}"
    )

with r6:

    st.write(
        f"**🤖 ML Readiness:** {ml_readiness}"
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

q1, q2, q3 = st.columns(3)

with q1:

    st.metric(
        "Missing Values",
        missing_values
    )

with q2:

    st.metric(
        "Duplicate Rows",
        duplicates
    )

with q3:

    st.metric(
        "Missing %",
        f"{missing_percentage:.2f}%"
    )


if missing_values == 0:

    st.success(
        "✅ No missing values detected."
    )

else:

    st.warning(
        f"⚠️ {missing_values} missing values detected."
    )


if duplicates == 0:

    st.success(
        "✅ No duplicate rows detected."
    )

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

    "Data Type":
        df.dtypes.astype(str).values,

    "Missing Values":
        df.isnull().sum().values,

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
            f"⚠️ {total_outliers} potential "
            "outlier values detected."
        )

        st.caption(
            f"Outlier burden: "
            f"{outlier_burden:.2f}% relative to dataset rows."
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

        st.write(
            f"🔒 `{column}`"
        )

else:

    st.success(
        "✅ No potentially sensitive column names detected."
    )

st.caption(
    "Privacy detection is based on column names "
    "and is a heuristic, not a guarantee."
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

    series = (
        pd.to_numeric(
            df[selected_column],
            errors="coerce"
        )
        .dropna()
    )

    if len(series) > 0:

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Minimum",
                f"{series.min():.2f}"
            )

        with s2:

            st.metric(
                "Maximum",
                f"{series.max():.2f}"
            )

        with s3:

            st.metric(
                "Mean",
                f"{series.mean():.2f}"
            )

        with s4:

            st.metric(
                "Median",
                f"{series.median():.2f}"
            )

        st.write("### Distribution")

        histogram_counts, bin_edges = np.histogram(
            series,
            bins=12
        )

        histogram_df = pd.DataFrame({

            "Range": [
                f"{bin_edges[i]:.2f} – "
                f"{bin_edges[i + 1]:.2f}"
                for i in range(
                    len(bin_edges) - 1
                )
            ],

            "Frequency": histogram_counts

        })

        st.bar_chart(
            histogram_df.set_index("Range"),
            y="Frequency",
            use_container_width=True
        )

        st.caption(
            f"Distribution of {selected_column} "
            f"across {len(series)} valid observations."
        )

    else:

        st.info(
            f"No valid numerical values found "
            f"in {selected_column}."
        )

else:

    st.info(
        "No numerical columns available for statistics."
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

recommendations = []

if missing_values > 0:

    recommendations.append(
        "Handle missing values before ML training."
    )

if duplicates > 0:

    recommendations.append(
        "Review and remove unnecessary duplicate records."
    )

if total_outliers > 0:

    recommendations.append(
        "Investigate detected outliers before model training."
    )

if sensitive_columns:

    recommendations.append(
        "Consider anonymizing or removing sensitive columns."
    )

if bias_available and bias_difference is not None:

    if bias_difference >= 20:

        recommendations.append(
            "Investigate the large outcome disparity between groups."
        )

    elif bias_difference >= 10:

        recommendations.append(
            "Review potential group-level outcome disparity."
        )

if not recommendations:

    recommendations.append(
        "Dataset shows no major automated quality issues."
    )

recommendations.append(
    "Perform exploratory data analysis before production ML use."
)


# =========================================================
# PROFESSIONAL REPORT TEXT
# =========================================================

report_text = f"""
DATAGUARDIAN AI
DATASET RISK REPORT

Dataset: {uploaded_file.name}

==================================================

DATASET SUMMARY

Rows: {rows}
Columns: {columns}
Numerical Columns: {len(numeric_columns)}
Categorical Columns: {len(categorical_columns)}

==================================================

DATA QUALITY

Missing Values: {missing_values}
Missing Percentage: {missing_percentage:.2f}%

Duplicate Rows: {duplicates}
Duplicate Percentage: {duplicate_percentage:.2f}%

==================================================

OUTLIER ANALYSIS

Potential Outliers: {total_outliers}
Outlier Burden: {outlier_burden:.2f}%
Outlier Risk: {outlier_status}

==================================================

PRIVACY ASSESSMENT

Privacy Risk: {privacy_risk}

Sensitive Columns:
{", ".join(map(str, sensitive_columns)) if sensitive_columns else "None detected"}

==================================================

BIAS ASSESSMENT

Bias Status: {bias_status}
"""

if bias_difference is not None:

    report_text += (
        f"Maximum Outcome Rate Difference: "
        f"{bias_difference:.2f}%\n"
    )

else:

    report_text += (
        "Maximum Outcome Rate Difference: "
        "Not available\n"
    )


report_text += f"""

==================================================

ML READINESS

{ml_readiness}

==================================================

FINAL HEALTH SCORE

{score:.1f}/100

Data Quality: {quality_status}
Outlier Risk: {outlier_status}
Privacy Risk: {privacy_display}
Bias Status: {bias_status}

==================================================

RECOMMENDATIONS

"""

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    report_text += (
        f"{i}. {recommendation}\n"
    )


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf_report():

    if not REPORTLAB_AVAILABLE:
        return None

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            "DataGuardian AI",
            title_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Dataset Risk Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            f"<b>Dataset:</b> {uploaded_file.name}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    summary_data = [

        ["Metric", "Result"],

        ["Rows", str(rows)],

        ["Columns", str(columns)],

        ["Missing Values", str(missing_values)],

        ["Duplicate Rows", str(duplicates)],

        ["Potential Outliers",
         str(total_outliers)],

        ["Privacy Risk", privacy_risk],

        ["Bias Status", bias_status],

        ["ML Readiness", ml_readiness],

        ["Overall Health",
         f"{score:.1f}/100"]

    ]

    table = Table(
        summary_data,
        colWidths=[220, 260]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1f2937")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    privacy_columns_text = (
        ", ".join(map(str, sensitive_columns))
        if sensitive_columns
        else "None detected"
    )

    bias_text = (
        f"Maximum outcome rate difference: "
        f"{bias_difference:.2f}%."
        if bias_difference is not None
        else "No binary outcome disparity was calculated."
    )

    sections = [

        (
            "Data Quality",
            f"Missing values: {missing_values} "
            f"({missing_percentage:.2f}%). "
            f"Duplicate rows: {duplicates}."
        ),

        (
            "Outlier Analysis",
            f"{total_outliers} potential outlier "
            f"values detected. "
            f"Outlier risk: {outlier_status}."
        ),

        (
            "Privacy Assessment",
            f"Privacy risk: {privacy_risk}. "
            f"Potentially sensitive columns: "
            f"{privacy_columns_text}."
        ),

        (
            "Bias Assessment",
            f"Bias status: {bias_status}. "
            f"{bias_text}"
        ),

        (
            "Machine Learning Readiness",
            ml_readiness
        )

    ]

    for heading, body in sections:

        story.append(
            Paragraph(
                heading,
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                body,
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(1, 12)
        )

    story.append(
        Paragraph(
            "Recommendations",
            styles["Heading2"]
        )
    )

    for recommendation in recommendations:

        story.append(
            Paragraph(
                f"• {recommendation}",
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(1, 5)
        )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Generated by DataGuardian AI",
            styles["Italic"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# PROFESSIONAL REPORT DOWNLOAD
# =========================================================

st.subheader("📥 Professional Report")

if REPORTLAB_AVAILABLE:

    pdf_report = create_pdf_report()

    st.download_button(
        label="📄 Download Professional PDF Report",
        data=pdf_report,
        file_name="DataGuardian_Dataset_Risk_Report.pdf",
        mime="application/pdf"
    )

else:

    st.warning(
        "PDF generation requires reportlab. "
        "Add reportlab to requirements.txt."
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
ML readiness: {ml_readiness}
"""

    # =====================================================
    # OPENAI ANALYSIS
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

Create a concise professional report with:

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

        report.append(
            ml_readiness
        )

        report.append(
            "## 💡 Recommendations"
        )

        for recommendation in recommendations:

            report.append(
                f"• {recommendation}"
            )

        ai_report = "\n\n".join(report)


    # =====================================================
    # SHOW AI REPORT
    # =====================================================

    st.markdown(ai_report)


    # =====================================================
    # TEXT DOWNLOAD
    # =====================================================

    full_report = (
        report_text
        +
        "\n\n"
        +
        ai_report
    )

    st.download_button(
        "📥 Download Text Analysis Report",
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
