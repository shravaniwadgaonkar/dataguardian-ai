import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DataGuardian AI")
st.write("Intelligent Dataset Health, Bias & Privacy Auditor")

st.divider()


# =====================================================
# OPENAI CONNECTION
# =====================================================

client = None

if OpenAI is not None and "OPENAI_API_KEY" in st.secrets:

    try:
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
    except Exception:
        client = None


# =====================================================
# UPLOAD DATASET
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        rows, columns = df.shape

        duplicates = int(
            df.duplicated().sum()
        )


        # =================================================
        # MISSING VALUES
        # =================================================

        missing_values = df.isnull().sum()

        total_missing = int(
            missing_values.sum()
        )


        # =================================================
        # NUMERICAL COLUMNS
        # =================================================

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns


        # =================================================
        # OUTLIER DETECTION
        # =================================================

        total_outliers = 0

        outlier_results = []

        for column in numeric_columns:

            data = df[column].dropna()

            if len(data) > 0:

                q1 = data.quantile(0.25)

                q3 = data.quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr

                upper = q3 + 1.5 * iqr

                outliers = data[
                    (data < lower) |
                    (data > upper)
                ]

                count = len(outliers)

                total_outliers += count

                outlier_results.append({

                    "Column": column,

                    "Outliers": count,

                    "Lower Bound": round(
                        lower, 2
                    ),

                    "Upper Bound": round(
                        upper, 2
                    )
                })


        # =================================================
        # PRIVACY AUDIT
        # =================================================

        sensitive_keywords = [

            "name",
            "email",
            "phone",
            "mobile",
            "address",
            "dob",
            "date_of_birth",
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

            column_name = str(
                column
            ).lower()

            if any(
                keyword in column_name
                for keyword in sensitive_keywords
            ):

                sensitive_columns.append(
                    column
                )


        if len(sensitive_columns) >= 3:

            privacy_risk = "HIGH"

        elif len(sensitive_columns) > 0:

            privacy_risk = "MEDIUM"

        else:

            privacy_risk = "LOW"


        # =================================================
        # REALISTIC HEALTH SCORE
        # =================================================

        score = 100.0


        # Missing value penalty

        if rows > 0 and columns > 0:

            missing_percentage = (

                total_missing /
                (rows * columns)

            ) * 100

            score -= min(
                missing_percentage * 2,
                30
            )


        # Duplicate penalty

        if rows > 0:

            duplicate_percentage = (

                duplicates /
                rows

            ) * 100

            score -= min(
                duplicate_percentage * 2,
                20
            )


        # Outlier penalty

        if len(numeric_columns) > 0:

            total_possible_values = (

                rows *
                len(numeric_columns)

            )

            if total_possible_values > 0:

                outlier_percentage = (

                    total_outliers /
                    total_possible_values

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


        # =================================================
        # DATASET OVERVIEW
        # =================================================

        st.subheader(
            "📊 Dataset Overview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                rows
            )

        with col2:

            st.metric(
                "Columns",
                columns
            )

        with col3:

            st.metric(
                "Duplicate Rows",
                duplicates
            )


        # =================================================
        # DATA QUALITY
        # =================================================

        st.subheader(
            "🔍 Data Quality"
        )

        if total_missing == 0:

            st.success(
                "✅ No missing values detected."
            )

        else:

            st.warning(
                f"⚠️ {total_missing} "
                "missing values detected."
            )


        if duplicates == 0:

            st.success(
                "✅ No duplicate rows detected."
            )

        else:

            st.warning(
                f"⚠️ {duplicates} "
                "duplicate rows detected."
            )


        # =================================================
        # COLUMN INFORMATION
        # =================================================

        st.subheader(
            "🧩 Column Information"
        )

        column_info = pd.DataFrame({

            "Column": df.columns,

            "Data Type":
                df.dtypes.astype(str),

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


        # =================================================
        # OUTLIER DETECTION
        # =================================================

        st.subheader(
            "🚨 Outlier Detection"
        )

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
                "No numerical columns available."
            )


        # =================================================
        # PRIVACY
        # =================================================

        st.subheader(
            "🔐 Privacy Risk Audit"
        )

        if sensitive_columns:

            st.warning(
                f"⚠️ {len(sensitive_columns)} "
                "potentially sensitive "
                "column(s) detected."
            )

            for column in sensitive_columns:

                st.write(
                    f"🔒 `{column}`"
                )

            st.metric(
                "Privacy Risk",
                privacy_risk
            )

        else:

            st.success(
                "✅ No potentially sensitive "
                "column names detected."
            )

        st.caption(
            "Privacy detection is based on "
            "column names and is a heuristic, "
            "not a guarantee."
        )


        # =================================================
        # BIAS AUDIT
        # =================================================

        st.subheader(
            "⚖️ Bias Audit"
        )

        categorical_columns = list(
            df.select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            ).columns
        )

        bias_available = False

        bias_disparity = None


        if categorical_columns:

            group_column = st.selectbox(

                "Group / Protected Attribute",

                categorical_columns

            )

            target_column = st.selectbox(

                "Outcome / Target",

                list(df.columns)

            )

            unique_targets = (

                df[target_column]
                .dropna()
                .unique()

            )


            if len(unique_targets) == 2:

                bias_available = True

                positive_value = st.selectbox(

                    "Positive Outcome",

                    unique_targets

                )


                temp = df[

                    [
                        group_column,
                        target_column
                    ]

                ].dropna()


                temp["positive"] = (

                    temp[target_column]
                    == positive_value

                )


                bias_table = (

                    temp
                    .groupby(group_column)
                    ["positive"]
                    .agg(
                        ["mean", "count"]
                    )
                    .reset_index()

                )


                bias_table[
                    "Outcome Rate (%)"
                ] = (

                    bias_table["mean"]
                    * 100

                ).round(2)


                bias_table = (
                    bias_table
                    .drop(columns=["mean"])
                )


                st.dataframe(

                    bias_table,

                    use_container_width=True

                )


                rates = bias_table[
                    "Outcome Rate (%)"
                ]


                if len(rates) >= 2:

                    bias_disparity = (

                        rates.max()
                        -
                        rates.min()

                    )


                    st.metric(

                        "Maximum Outcome "
                        "Rate Difference",

                        f"{bias_disparity:.2f}%"

                    )


                    if bias_disparity >= 20:

                        st.error(
                            "🔴 Large outcome "
                            "disparity detected."
                        )

                    elif bias_disparity >= 10:

                        st.warning(
                            "🟡 Potential outcome "
                            "disparity detected."
                        )

                    else:

                        st.success(
                            "🟢 No large outcome "
                            "disparity detected."
                        )


                    st.caption(
                        "A disparity does not "
                        "prove discrimination."
                    )


            else:

                st.info(
                    "Bias analysis currently "
                    "supports binary outcomes only."
                )


        else:

            st.info(
                "No categorical columns found. "
                "Bias analysis cannot be performed "
                "automatically."
            )


        # =================================================
        # RISK REPORT
        # =================================================

        st.subheader(
            "🛡️ Dataset Risk Report"
        )


        if score >= 85:

            quality_status = (
                "🟢 Excellent"
            )

        elif score >= 70:

            quality_status = (
                "🟡 Good"
            )

        elif score >= 50:

            quality_status = (
                "🟠 Needs Attention"
            )

        else:

            quality_status = (
                "🔴 Poor"
            )


        if total_outliers == 0:

            outlier_status = (
                "🟢 Low"
            )

        elif total_outliers < (
            rows * 0.10
        ):

            outlier_status = (
                "🟡 Moderate"
            )

        else:

            outlier_status = (
                "🔴 High"
            )


        if bias_available:

            bias_status = (
                "🟢 Evaluated"
            )

        else:

            bias_status = (
                "⚪ Not Evaluated"
            )


        st.write(
            f"**Overall Health:** "
            f"{score:.1f}/100"
        )

        st.write(
            f"**Data Quality:** "
            f"{quality_status}"
        )

        st.write(
            f"**Outlier Risk:** "
            f"{outlier_status}"
        )

        st.write(
            f"**Privacy Risk:** "
            f"{privacy_risk}"
        )

        st.write(
            f"**Bias Status:** "
            f"{bias_status}"
        )


        # =================================================
        # STATISTICS
        # =================================================

        st.subheader(
            "📈 Dataset Statistics"
        )


        if len(numeric_columns) > 0:

            selected_column = st.selectbox(

                "Select numerical column",

                numeric_columns,

                key="chart_column"

            )


            chart_data = (
                df[selected_column]
                .dropna()
            )


            fig, ax = plt.subplots()

            ax.hist(
                chart_data,
                bins=20
            )

            ax.set_title(
                f"Distribution of "
                f"{selected_column}"
            )

            ax.set_xlabel(
                selected_column
            )

            ax.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig)

            plt.close(fig)


        # =================================================
        # AI DATA ANALYST
        # =================================================

        st.subheader(
            "🤖 AI Data Analyst"
        )

        st.write(
            "Generate an intelligent report "
            "about your dataset."
        )


        if st.button(
            "🤖 Generate AI Analysis",
            type="primary"
        ):


            dataset_summary = f"""

Dataset: {uploaded_file.name}

Rows: {rows}

Columns: {columns}

Missing values: {total_missing}

Duplicate rows: {duplicates}

Potential outliers: {total_outliers}

Sensitive columns:
{sensitive_columns}

Privacy risk:
{privacy_risk}

Health score:
{score:.1f}/100

Bias status:
{bias_status}

"""


            ai_report = None


            # -----------------------------------------
            # REAL AI
            # -----------------------------------------

            if client is not None:

                try:

                    with st.spinner(
                        "🤖 AI is analyzing "
                        "your dataset..."
                    ):

                        response = (
                            client.responses.create(

                                model="gpt-5-mini",

                                input=f"""

You are a professional "
Data Scientist.

Analyze this dataset summary:

{dataset_summary}

Create a concise professional report.

Use these sections:

## Dataset Overview

## Data Quality Findings

## Outlier Analysis

## Privacy Considerations

## Bias Considerations

## Machine Learning Readiness

## Recommended Actions

Do not invent information.
"""

                            )
                        )


                    ai_report = (
                        response.output_text
                    )


                except Exception:

                    ai_report = None


            # -----------------------------------------
            # FALLBACK
            # -----------------------------------------

            if ai_report is None:

                st.info(
                    "ℹ️ Real AI is currently "
                    "unavailable. DataGuardian's "
                    "local analysis engine is "
                    "generating the report."
                )


                report = []


                report.append(
                    "## 📋 Dataset Overview"
                )


                report.append(

                    f"The dataset contains "
                    f"**{rows} rows** and "
                    f"**{columns} columns**."

                )


                report.append(
                    "## 🔍 Data Quality"
                )


                if total_missing == 0:

                    report.append(
                        "✅ No missing values "
                        "were detected."
                    )

                else:

                    report.append(

                        f"⚠️ {total_missing} "
                        "missing values were detected."

                    )


                if duplicates == 0:

                    report.append(
                        "✅ No duplicate "
                        "records were detected."
                    )

                else:

                    report.append(

                        f"⚠️ {duplicates} "
                        "duplicate records were detected."

                    )


                report.append(
                    "## 🚨 Outlier Analysis"
                )


                if total_outliers > 0:

                    report.append(

                        f"⚠️ **{total_outliers} "
                        "potential outlier values** "
                        "were detected."

                    )

                    report.append(

                        "These values should be "
                        "investigated before "
                        "machine-learning training."

                    )

                else:

                    report.append(
                        "✅ No potential "
                        "outliers were detected."
                    )


                report.append(
                    "## 🔐 Privacy"
                )


                if sensitive_columns:

                    report.append(

                        "Potentially sensitive "
                        "columns detected: "
                        +
                        ", ".join(
                            map(
                                str,
                                sensitive_columns
                            )
                        )

                    )

                else:

                    report.append(
                        "✅ No potentially "
                        "sensitive column names "
                        "were detected."
                    )


                report.append(
                    "## ⚖️ Bias"
                )


                if bias_available:

                    report.append(
                        "Bias analysis was "
                        "performed using the "
                        "available categorical data."
                    )

                else:

                    report.append(
                        "Bias analysis could not "
                        "be performed because "
                        "no suitable categorical "
                        "group column was detected."
                    )


                report.append(
                    "## 🤖 Machine Learning Readiness"
                )


                if score >= 85:

                    report.append(
                        "The dataset appears "
                        "to be in good condition "
                        "for initial ML experimentation."
                    )

                elif score >= 70:

                    report.append(
                        "The dataset may be "
                        "usable for ML, but "
                        "some quality checks "
                        "should be completed first."
                    )

                else:

                    report.append(
                        "The dataset requires "
                        "additional preprocessing "
                        "before ML training."
                    )


                report.append(
                    "## 💡 Recommended Actions"
                )


                if total_outliers > 0:

                    report.append(
                        "• Investigate potential "
                        "outliers."
                    )


                if total_missing > 0:

                    report.append(
                        "• Handle missing values."
                    )


                if duplicates > 0:

                    report.append(
                        "• Review duplicate records."
                    )


                if sensitive_columns:

                    report.append(
                        "• Consider anonymizing "
                        "sensitive information."
                    )


                report.append(
                    "• Perform exploratory "
                    "data analysis before "
                    "machine-learning training."
                )


                ai_report = (
                    "\n\n".join(report)
                )


            st.markdown(
                ai_report
            )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            report_text = f"""

DATAGUARDIAN AI
DATASET ANALYSIS REPORT

Dataset: {uploaded_file.name}

Rows: {rows}

Columns: {columns}

Missing Values: {total_missing}

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

                report_text,

                file_name=(
                    "DataGuardian_Report.txt"
                ),

                mime="text/plain"

            )


        # =================================================
        # PREVIEW
        # =================================================

        st.subheader(
            "👀 Dataset Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # =================================================
        # FINAL HEALTH SCORE
        # =================================================

        st.subheader(
            "🏥 Dataset Health Score"
        )

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
                "🟡 Dataset is usable but "
                "needs some attention."
            )

        elif score >= 50:

            st.warning(
                "🟠 Dataset requires cleaning "
                "before ML use."
            )

        else:

            st.error(
                "🔴 Dataset requires significant "
                "preprocessing."
            )


    except Exception as e:

        st.error(
            f"Unable to read the dataset: {e}"
        )
