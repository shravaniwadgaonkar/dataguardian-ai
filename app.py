import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from google import genai

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DataGuardian AI")
st.caption(
    "AI Dataset Health, Risk & Intelligent Cleaning Platform"
)

st.divider()


# =========================================================
# UPLOAD DATASET
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV dataset to begin.")
    st.stop()

try:
    original_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

# Never modify the original dataset
df = original_df.copy()

st.success("✅ Dataset uploaded successfully!")


# =========================================================
# FUNCTIONS
# =========================================================

def calculate_outliers(dataframe):

    total = 0
    details = []

    numeric_cols = dataframe.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        series = dataframe[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(
                ((series < lower) |
                 (series > upper)).sum()
            )

        total += count

        details.append({
            "Column": col,
            "Outliers": count
        })

    return total, details


def dataset_metrics(dataframe):

    rows, cols = dataframe.shape

    missing = int(
        dataframe.isnull().sum().sum()
    )

    duplicates = int(
        dataframe.duplicated().sum()
    )

    outliers, details = calculate_outliers(
        dataframe
    )

    return {
        "rows": rows,
        "columns": cols,
        "missing": missing,
        "duplicates": duplicates,
        "outliers": outliers,
        "outlier_details": details
    }


def create_health_score(metrics):

    rows = metrics["rows"]

    if rows == 0:
        return 0

    cells = (
        metrics["rows"] *
        metrics["columns"]
    )

    missing_pct = (
        metrics["missing"] /
        cells * 100
        if cells else 0
    )

    duplicate_pct = (
        metrics["duplicates"] /
        rows * 100
    )

    outlier_pct = (
        metrics["outliers"] /
        rows * 100
    )

    score = 100

    score -= min(
        missing_pct * 2,
        25
    )

    score -= min(
        duplicate_pct * 2,
        15
    )

    score -= min(
        outlier_pct * 0.8,
        40
    )

    return round(
        max(0, min(100, score)),
        1
    )


def fill_missing_values(dataframe):

    result = dataframe.copy()

    changes = []

    for col in result.columns:

        missing = int(
            result[col].isnull().sum()
        )

        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(
            result[col]
        ):

            value = result[col].median()

            result[col] = result[col].fillna(
                value
            )

            changes.append(
                f"{col}: {missing} missing values "
                f"filled with median ({value:.2f})"
            )

        else:

            mode = result[col].mode()

            if len(mode) > 0:

                value = mode.iloc[0]

                result[col] = result[col].fillna(
                    value
                )

                changes.append(
                    f"{col}: {missing} missing values "
                    f"filled with most frequent value "
                    f"'{value}'"
                )

            else:

                result[col] = result[col].fillna(
                    "Unknown"
                )

                changes.append(
                    f"{col}: {missing} missing values "
                    "filled with 'Unknown'"
                )

    return result, changes


def remove_duplicates(dataframe):

    result = dataframe.copy()

    before = len(result)

    result = result.drop_duplicates()

    removed = before - len(result)

    return result, removed


def cap_outliers(dataframe):

    result = dataframe.copy()

    changes = []

    numeric_cols = result.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        series = result[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (result[col] < lower) |
            (result[col] > upper)
        )

        count = int(mask.sum())

        if count > 0:

            result[col] = result[col].clip(
                lower,
                upper
            )

            changes.append(
                f"{col}: capped {count} "
                "potential outliers using IQR limits"
            )

    return result, changes


# =========================================================
# INITIAL AUDIT
# =========================================================

before = dataset_metrics(df)
before_score = create_health_score(before)

st.subheader("📊 Dataset Overview")

a, b, c, d = st.columns(4)

with a:
    st.metric(
        "Rows",
        before["rows"]
    )

with b:
    st.metric(
        "Columns",
        before["columns"]
    )

with c:
    st.metric(
        "Missing Values",
        before["missing"]
    )

with d:
    st.metric(
        "Duplicates",
        before["duplicates"]
    )


# =========================================================
# HEALTH SCORE
# =========================================================

st.subheader("🏥 Current Dataset Health")

st.metric(
    "Health Score",
    f"{before_score}/100"
)

if before_score >= 85:
    st.success(
        "🟢 Dataset health is excellent."
    )
elif before_score >= 70:
    st.warning(
        "🟡 Dataset health is good but needs attention."
    )
elif before_score >= 50:
    st.warning(
        "🟠 Dataset requires preprocessing."
    )
else:
    st.error(
        "🔴 Dataset requires significant cleaning."
    )


# =========================================================
# CLEANING OPTIONS
# =========================================================

st.divider()

st.subheader("🧹 AI Data Cleaning Agent")

st.write(
    "Choose the cleaning operations DataGuardian "
    "should perform. Your original uploaded dataset "
    "will never be modified."
)

fix_missing = st.checkbox(
    "🔧 Handle missing values",
    value=True
)

fix_duplicates = st.checkbox(
    "🗑️ Remove duplicate rows",
    value=True
)

fix_outliers = st.checkbox(
    "📉 Treat numerical outliers",
    value=True
)


# =========================================================
# AI CLEANING
# =========================================================

if st.button(
    "🤖 Run AI Data Cleaning Agent",
    type="primary"
):

    cleaned_df = df.copy()

    cleaning_log = []

    # -------------------------
    # Missing Values
    # -------------------------

    if fix_missing:

        cleaned_df, changes = (
            fill_missing_values(
                cleaned_df
            )
        )

        cleaning_log.extend(changes)


    # -------------------------
    # Duplicates
    # -------------------------

    if fix_duplicates:

        cleaned_df, removed = (
            remove_duplicates(
                cleaned_df
            )
        )

        if removed > 0:

            cleaning_log.append(
                f"Removed {removed} duplicate rows."
            )

        else:

            cleaning_log.append(
                "No duplicate rows required removal."
            )


    # -------------------------
    # Outliers
    # -------------------------

    if fix_outliers:

        cleaned_df, changes = (
            cap_outliers(
                cleaned_df
            )
        )

        cleaning_log.extend(changes)


    # =====================================================
    # AFTER AUDIT
    # =====================================================

    after = dataset_metrics(
        cleaned_df
    )

    after_score = create_health_score(
        after
    )


    # =====================================================
    # SAVE IN SESSION
    # =====================================================

    st.session_state["cleaned_df"] = (
        cleaned_df
    )

    st.session_state["before"] = before
    st.session_state["after"] = after

    st.session_state["before_score"] = (
        before_score
    )

    st.session_state["after_score"] = (
        after_score
    )

    st.session_state["cleaning_log"] = (
        cleaning_log
    )

    st.success(
        "✅ AI Data Cleaning Agent completed."
    )


# =========================================================
# SHOW RESULTS
# =========================================================

if "cleaned_df" in st.session_state:

    cleaned_df = st.session_state[
        "cleaned_df"
    ]

    before = st.session_state[
        "before"
    ]

    after = st.session_state[
        "after"
    ]

    before_score = st.session_state[
        "before_score"
    ]

    after_score = st.session_state[
        "after_score"
    ]

    cleaning_log = st.session_state[
        "cleaning_log"
    ]


    # =====================================================
    # BEFORE / AFTER
    # =====================================================

    st.divider()

    st.subheader(
        "📈 Before vs After"
    )

    x1, x2 = st.columns(2)

    with x1:

        st.markdown(
            "### 🔴 Before Cleaning"
        )

        st.metric(
            "Missing",
            before["missing"]
        )

        st.metric(
            "Duplicates",
            before["duplicates"]
        )

        st.metric(
            "Outliers",
            before["outliers"]
        )

        st.metric(
            "Health",
            f"{before_score}/100"
        )


    with x2:

        st.markdown(
            "### 🟢 After Cleaning"
        )

        st.metric(
            "Missing",
            after["missing"]
        )

        st.metric(
            "Duplicates",
            after["duplicates"]
        )

        st.metric(
            "Outliers",
            after["outliers"]
        )

        st.metric(
            "Health",
            f"{after_score}/100"
        )


    # =====================================================
    # HEALTH IMPROVEMENT
    # =====================================================

    improvement = (
        after_score -
        before_score
    )

    if improvement > 0:

        st.success(
            f"📈 Dataset health improved by "
            f"{improvement:.1f} points."
        )

    elif improvement == 0:

        st.info(
            "Dataset health score remained unchanged."
        )

    else:

        st.warning(
            f"Health score changed by "
            f"{improvement:.1f} points."
        )


    # =====================================================
    # CLEANING LOG
    # =====================================================

    st.subheader(
        "📝 Cleaning Actions"
    )

    if cleaning_log:

        for item in cleaning_log:

            st.write(
                f"✅ {item}"
            )

    else:

        st.info(
            "No cleaning actions were required."
        )


    # =====================================================
    # GEMINI EXPLANATION
    # =====================================================

    st.divider()

    st.subheader(
        "✨ Gemini Cleaning Explanation"
    )

    if st.button(
        "✨ Ask Gemini to Explain the Cleaning"
    ):

        if "GEMINI_API_KEY" not in st.secrets:

            st.error(
                "GEMINI_API_KEY is missing. "
                "Add it in Streamlit Secrets."
            )

        else:

            try:

                client = genai.Client(
                    api_key=st.secrets[
                        "GEMINI_API_KEY"
                    ]
                )

                prompt = f"""
You are DataGuardian AI,
a professional data governance assistant.

A dataset was cleaned by an automated
data cleaning pipeline.

BEFORE:

Rows: {before["rows"]}
Columns: {before["columns"]}
Missing values: {before["missing"]}
Duplicates: {before["duplicates"]}
Potential outliers: {before["outliers"]}
Health score: {before_score}/100

AFTER:

Rows: {after["rows"]}
Columns: {after["columns"]}
Missing values: {after["missing"]}
Duplicates: {after["duplicates"]}
Potential outliers: {after["outliers"]}
Health score: {after_score}/100

CLEANING ACTIONS:

{cleaning_log}

Explain:

1. What changed?
2. Why were these cleaning operations reasonable?
3. What risks remain?
4. Is the dataset more suitable for ML now?
5. What should a data scientist inspect manually?

Do not invent information.
Clearly distinguish automated cleaning
from recommendations requiring human review.
"""

                with st.spinner(
                    "Gemini is reviewing the cleaning..."
                ):

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt
                    )

                st.success(
                    "✅ Gemini explanation generated."
                )

                st.markdown(
                    response.text
                )

                st.session_state[
                    "gemini_cleaning_report"
                ] = response.text

            except Exception as e:

                st.error(
                    "Gemini analysis failed."
                )

                st.code(
                    str(e)
                )


    # =====================================================
    # DOWNLOAD CLEANED DATASET
    # =====================================================

    st.divider()

    st.subheader(
        "📥 Download Cleaned Dataset"
    )

    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download cleaned_dataset.csv",
        csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )


    # =====================================================
    # CLEANED PREVIEW
    # =====================================================

    st.subheader(
        "👀 Cleaned Dataset Preview"
    )

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True
    )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    st.divider()

    st.subheader(
        "📋 Cleaning Summary"
    )

    report = f"""
DATAGUARDIAN AI
AI DATA CLEANING REPORT

Original Dataset
----------------
Rows: {before["rows"]}
Columns: {before["columns"]}
Missing Values: {before["missing"]}
Duplicates: {before["duplicates"]}
Potential Outliers: {before["outliers"]}
Health Score: {before_score}/100

Cleaned Dataset
---------------
Rows: {after["rows"]}
Columns: {after["columns"]}
Missing Values: {after["missing"]}
Duplicates: {after["duplicates"]}
Potential Outliers: {after["outliers"]}
Health Score: {after_score}/100

Health Improvement:
{improvement:.1f} points

Cleaning Actions:
"""

    for item in cleaning_log:

        report += (
            f"\n- {item}"
        )


    if (
        "gemini_cleaning_report"
        in st.session_state
    ):

        report += """

GEMINI REVIEW
=============

"""

        report += (
            st.session_state[
                "gemini_cleaning_report"
            ]
        )


    st.download_button(
        "📄 Download Cleaning Report",
        report,
        file_name="DataGuardian_Cleaning_Report.txt",
        mime="text/plain"
    )


# =========================================================
# ORIGINAL DATA PREVIEW
# =========================================================

st.divider()

st.subheader(
    "👀 Original Dataset Preview"
)

st.dataframe(
    original_df.head(10),
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ DataGuardian AI — AI-assisted dataset "
    "quality, privacy, bias and machine-learning "
    "readiness auditing."
)
