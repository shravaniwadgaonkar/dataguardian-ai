import streamlit as st
import pandas as pd
from io import BytesIO
from google import genai

# Optional PDF support
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB = True
except ImportError:
    REPORTLAB = False


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DataGuardian AI")
st.caption(
    "Intelligent Dataset Health, Bias & Privacy Auditor"
)

st.divider()


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV dataset to start the audit.")
    st.stop()


try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()


st.success("✅ Dataset uploaded successfully!")


# =========================================================
# BASIC DATASET INFORMATION
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
        include=["object", "category", "bool"]
    ).columns
)


# =========================================================
# OUTLIER DETECTION
# =========================================================

total_outliers = 0
outlier_details = []

for column in numeric_columns:

    data = df[column].dropna()

    if len(data) < 4:
        continue

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((data < lower) | (data > upper)).sum()
        )

    total_outliers += count

    outlier_details.append({
        "Column": column,
        "Outliers": count
    })


# =========================================================
# PRIVACY AUDIT
# =========================================================

privacy_keywords = [
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
    "postal",
    "ssn"
]

sensitive_columns = []

for column in df.columns:

    name = str(column).lower()

    if any(
        keyword in name
        for keyword in privacy_keywords
    ):
        sensitive_columns.append(column)


if len(sensitive_columns) >= 3:
    privacy_risk = "HIGH"
elif len(sensitive_columns) > 0:
    privacy_risk = "MEDIUM"
else:
    privacy_risk = "LOW"


# =========================================================
# METRICS
# =========================================================

total_cells = rows * columns

missing_percentage = (
    (missing_values / total_cells) * 100
    if total_cells > 0 else 0
)

duplicate_percentage = (
    (duplicates / rows) * 100
    if rows > 0 else 0
)

outlier_percentage = (
    (total_outliers / rows) * 100
    if rows > 0 else 0
)


# =========================================================
# HEALTH SCORE
# =========================================================

score = 100.0

score -= min(
    missing_percentage * 2,
    25
)

score -= min(
    duplicate_percentage * 2,
    15
)

score -= min(
    outlier_percentage * 0.8,
    40
)

if privacy_risk == "MEDIUM":
    score -= 7

elif privacy_risk == "HIGH":
    score -= 15


score = max(
    0,
    min(100, score)
)


# =========================================================
# STATUS
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
elif outlier_percentage < 5:
    outlier_status = "🟡 Moderate"
else:
    outlier_status = "🔴 High"


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
    st.metric("Duplicate Rows", duplicates)


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
        "Missing %",
        f"{missing_percentage:.2f}%"
    )

with q3:
    st.metric(
        "Duplicate %",
        f"{duplicate_percentage:.2f}%"
    )


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
    "Data Type": [
        str(df[c].dtype)
        for c in df.columns
    ],
    "Missing": [
        int(df[c].isnull().sum())
        for c in df.columns
    ],
    "Unique": [
        int(df[c].nunique())
        for c in df.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True
)


# =========================================================
# OUTLIERS
# =========================================================

st.subheader("🚨 Outlier Detection")

if outlier_details:

    outlier_df = pd.DataFrame(
        outlier_details
    )

    st.dataframe(
        outlier_df,
        use_container_width=True
    )

    if total_outliers > 0:
        st.warning(
            f"⚠️ {total_outliers} potential "
            "outlier values detected."
        )
    else:
        st.success(
            "✅ No potential outliers detected."
        )

else:

    st.info(
        "No numerical columns available for "
        "outlier detection."
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

bias_status = "⚪ Not Evaluated"
bias_difference = None
bias_info = "Bias analysis unavailable."

if categorical_columns:

    group_column = st.selectbox(
        "Select group column",
        categorical_columns
    )

    possible_targets = [
        c for c in df.columns
        if c != group_column
    ]

    target_column = st.selectbox(
        "Select outcome column",
        possible_targets
    )

    unique_values = (
        df[target_column]
        .dropna()
        .unique()
    )

    if len(unique_values) == 2:

        positive_value = st.selectbox(
            "Select positive outcome",
            unique_values
        )

        temp = df[
            [group_column, target_column]
        ].dropna().copy()

        temp["positive"] = (
            temp[target_column]
            == positive_value
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

            bias_status = "🟢 Evaluated"

            if bias_difference >= 20:

                bias_info = (
                    "Large outcome disparity detected."
                )

                st.error(
                    f"🔴 Outcome difference: "
                    f"{bias_difference:.2f}%"
                )

            elif bias_difference >= 10:

                bias_info = (
                    "Potential outcome disparity detected."
                )

                st.warning(
                    f"🟡 Outcome difference: "
                    f"{bias_difference:.2f}%"
                )

            else:

                bias_info = (
                    "No large outcome disparity detected."
                )

                st.success(
                    f"🟢 Outcome difference: "
                    f"{bias_difference:.2f}%"
                )

    else:

        st.info(
            "Select a binary outcome column "
            "for automatic bias analysis."
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

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Health",
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
        privacy_risk
    )

st.write(
    f"**⚖️ Bias Status:** {bias_status}"
)


# =========================================================
# ML READINESS
# =========================================================

if score >= 85:
    ml_readiness = (
        "🟢 Ready for initial ML experiments"
    )
elif score >= 70:
    ml_readiness = (
        "🟡 Usable after preprocessing"
    )
elif score >= 50:
    ml_readiness = (
        "🟠 Needs significant preprocessing"
    )
else:
    ml_readiness = (
        "🔴 Requires significant cleaning"
    )

st.write(
    f"**🤖 ML Readiness:** {ml_readiness}"
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
        "Review, anonymize or remove sensitive columns."
    )

if bias_difference is not None and bias_difference >= 10:
    recommendations.append(
        "Investigate group-level outcome disparities."
    )

recommendations.append(
    "Perform exploratory data analysis before production ML use."
)

st.subheader("💡 Recommendations")

for item in recommendations:
    st.write(f"• {item}")


# =========================================================
# GOOGLE GEMINI AI
# =========================================================

st.divider()

st.subheader("✨ Google Gemini AI Analyst")

st.write(
    "Gemini interprets DataGuardian's measured audit "
    "results and generates an explainable AI report."
)


if st.button(
    "✨ Analyze Dataset with Gemini",
    type="primary"
):

    if "GEMINI_API_KEY" not in st.secrets:

        st.error(
            "GEMINI_API_KEY is missing. "
            "Add it in Streamlit → Manage app → Settings → Secrets."
        )

    else:

        try:

            client = genai.Client(
                api_key=st.secrets[
                    "GEMINI_API_KEY"
                ]
            )

            audit_data = f"""
DATASET AUDIT RESULTS

Dataset: {uploaded_file.name}

Rows: {rows}
Columns: {columns}

Missing Values: {missing_values}
Missing Percentage: {missing_percentage:.2f}%

Duplicate Rows: {duplicates}
Duplicate Percentage: {duplicate_percentage:.2f}%

Potential Outliers: {total_outliers}
Outlier Percentage: {outlier_percentage:.2f}%

Data Quality: {quality_status}
Outlier Risk: {outlier_status}

Privacy Risk: {privacy_risk}

Sensitive Columns:
{sensitive_columns}

Bias Status: {bias_status}

Bias Finding:
{bias_info}

Bias Difference:
{bias_difference}

Overall Health Score:
{score:.1f}/100

Machine Learning Readiness:
{ml_readiness}

Recommendations:
{recommendations}
"""

            prompt = f"""
You are DataGuardian AI,
an expert dataset governance analyst.

Analyze ONLY the evidence provided below.

Do not invent facts.

Separate measured findings from your interpretation.

{audit_data}

Create a professional report with:

## 🧠 Executive Summary

## 🚨 Key Risks

## 🔍 Data Quality Analysis

## 📊 Outlier Analysis

## 🔐 Privacy Analysis

## ⚖️ Bias Analysis

## 🤖 Machine Learning Readiness

## 💡 Priority Recommendations

Give exactly 5 practical recommendations.

For each recommendation include:
- Priority
- Action
- Reason

## 🏁 Final Risk Level

Choose exactly one:

LOW
MEDIUM
HIGH
CRITICAL

Explain the decision.

Important:
Privacy detection is based on column names and
does NOT prove that personal information is absent.

Bias analysis may be incomplete if suitable
group and outcome data is unavailable.
"""

            with st.spinner(
                "✨ Gemini is analyzing your dataset..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

            st.success(
                "✅ Gemini analysis completed."
            )

            st.markdown(
                response.text
            )

            gemini_report = f"""
DATAGUARDIAN AI
GOOGLE GEMINI ANALYSIS REPORT

Dataset: {uploaded_file.name}

Health Score: {score:.1f}/100

Data Quality: {quality_status}

Outlier Risk: {outlier_status}

Privacy Risk: {privacy_risk}

Bias Status: {bias_status}

ML Readiness: {ml_readiness}

================================

GEMINI ANALYSIS

{response.text}
"""

            st.download_button(
                "📥 Download Gemini Report",
                gemini_report,
                file_name="DataGuardian_Gemini_Report.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(
                "Gemini analysis failed."
            )

            st.code(
                str(e)
            )


# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# STATISTICS
# =========================================================

st.subheader("📈 Dataset Statistics")

if numeric_columns:

    selected_column = st.selectbox(
        "Select numerical column",
        numeric_columns
    )

    series = df[
        selected_column
    ].dropna()

    if len(series) > 0:

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Minimum",
                f"{series.min():.2f}"
            )

        with b:
            st.metric(
                "Maximum",
                f"{series.max():.2f}"
            )

        with c:
            st.metric(
                "Mean",
                f"{series.mean():.2f}"
            )

        with d:
            st.metric(
                "Median",
                f"{series.median():.2f}"
            )

        histogram = pd.DataFrame(
            {
                "Value": series
            }
        )

        st.bar_chart(
            histogram["Value"]
            .value_counts()
            .sort_index()
        )

else:

    st.info(
        "No numerical columns available."
    )


# =========================================================
# TEXT REPORT
# =========================================================

text_report = f"""
DATAGUARDIAN AI
DATASET RISK REPORT

Dataset:
{uploaded_file.name}

Rows:
{rows}

Columns:
{columns}

Missing Values:
{missing_values}

Duplicate Rows:
{duplicates}

Potential Outliers:
{total_outliers}

Privacy Risk:
{privacy_risk}

Bias Status:
{bias_status}

ML Readiness:
{ml_readiness}

Overall Health:
{score:.1f}/100

RECOMMENDATIONS

"""

for i, item in enumerate(
    recommendations,
    1
):

    text_report += (
        f"{i}. {item}\n"
    )


st.download_button(
    "📄 Download Text Report",
    text_report,
    file_name="DataGuardian_Report.txt",
    mime="text/plain"
)


# =========================================================
# PDF REPORT
# =========================================================

if REPORTLAB:

    def create_pdf():

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "DataGuardian AI",
                styles["Title"]
            )
        )

        story.append(
            Paragraph(
                "Dataset Risk Report",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(1, 15)
        )

        report_lines = [
            f"Dataset: {uploaded_file.name}",
            f"Rows: {rows}",
            f"Columns: {columns}",
            f"Missing Values: {missing_values}",
            f"Duplicate Rows: {duplicates}",
            f"Potential Outliers: {total_outliers}",
            f"Privacy Risk: {privacy_risk}",
            f"Bias Status: {bias_status}",
            f"ML Readiness: {ml_readiness}",
            f"Health Score: {score:.1f}/100"
        ]

        for line in report_lines:

            story.append(
                Paragraph(
                    line,
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                "Recommendations",
                styles["Heading2"]
            )
        )

        for item in recommendations:

            story.append(
                Paragraph(
                    "• " + item,
                    styles["BodyText"]
                )
            )

        doc.build(story)

        buffer.seek(0)

        return buffer.getvalue()


    pdf_data = create_pdf()

    st.download_button(
        "📄 Download Professional PDF Report",
        pdf_data,
        file_name="DataGuardian_Risk_Report.pdf",
        mime="application/pdf"
    )


# =========================================================
# FINAL HEALTH SCORE
# =========================================================

st.divider()

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
        "🟡 Dataset health is good but needs some attention."
    )

elif score >= 50:

    st.warning(
        "🟠 Dataset requires preprocessing."
    )

else:

    st.error(
        "🔴 Dataset requires significant cleaning."
    )
