import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="DataGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DataGuardian AI")
st.write("Intelligent Dataset Health, Bias & Privacy Auditor")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        rows, columns = df.shape
        duplicates = df.duplicated().sum()

        # =========================
        # DATASET OVERVIEW
        # =========================

        st.subheader("📊 Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", rows)

        with col2:
            st.metric("Columns", columns)

        with col3:
            st.metric("Duplicate Rows", duplicates)

        st.divider()

        # =========================
        # DATA QUALITY
        # =========================

        st.subheader("🔍 Data Quality")

        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing == 0:
            st.success("✅ No missing values detected.")
        else:
            st.warning(
                f"⚠️ {total_missing} missing values detected."
            )

            missing_table = pd.DataFrame({
                "Column": missing_values.index,
                "Missing Values": missing_values.values
            })

            missing_table = missing_table[
                missing_table["Missing Values"] > 0
            ]

            st.dataframe(
                missing_table,
                use_container_width=True
            )

        # =========================
        # DUPLICATES
        # =========================

        if duplicates > 0:
            st.warning(
                f"⚠️ {duplicates} duplicate rows detected."
            )
        else:
            st.success("✅ No duplicate rows detected.")

        # =========================
        # COLUMN INFORMATION
        # =========================

        st.subheader("🧩 Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
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

        # =========================
        # OUTLIER DETECTION
        # =========================

        st.subheader("🚨 Outlier Detection")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        total_outliers = 0

        if len(numeric_columns) == 0:

            st.info("No numerical columns found.")

        else:

            outlier_results = []

            for column in numeric_columns:

                data = df[column].dropna()

                if len(data) > 0:

                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)

                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = data[
                        (data < lower_bound) |
                        (data > upper_bound)
                    ]

                    outlier_count = len(outliers)

                    total_outliers += outlier_count

                    outlier_results.append({
                        "Column": column,
                        "Outliers": outlier_count,
                        "Lower Bound": round(
                            lower_bound, 2
                        ),
                        "Upper Bound": round(
                            upper_bound, 2
                        )
                    })

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
            else:
                st.success(
                    "✅ No potential outliers detected."
                )

        # =========================
        # PRIVACY RISK AUDIT
        # =========================

        st.subheader("🔐 Privacy Risk Audit")

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

            column_name = str(column).lower()

            for keyword in sensitive_keywords:

                if keyword in column_name:
                    sensitive_columns.append(column)
                    break

        if sensitive_columns:

            st.warning(
                f"⚠️ {len(sensitive_columns)} potentially "
                "sensitive column(s) detected."
            )

            for column in sensitive_columns:
                st.write(f"🔒 `{column}`")

            if len(sensitive_columns) >= 3:
                privacy_risk = "HIGH"
            else:
                privacy_risk = "MEDIUM"

            st.metric(
                "Privacy Risk",
                privacy_risk
            )

            st.info(
                "Recommendation: Remove, mask, or anonymize "
                "sensitive information before sharing the dataset."
            )

        else:

            st.success(
                "✅ No potentially sensitive column names detected."
            )

        st.caption(
            "Privacy detection is based on column names. "
            "It is a heuristic and does not guarantee that "
            "personal information is absent."
        )

        # =========================
        # DATASET PREVIEW
        # =========================

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # =========================
        # DATASET HEALTH SCORE
        # =========================

        st.subheader("🏥 Dataset Health Score")

        score = 100.0

        if total_missing > 0:

            missing_percentage = (
                total_missing /
                (rows * columns)
            ) * 100

            score -= min(
                missing_percentage * 2,
                30
            )

        if duplicates > 0:

            duplicate_percentage = (
                duplicates / rows
            ) * 100

            score -= min(
                duplicate_percentage,
                20
            )

        if rows > 0:

            outlier_percentage = (
                total_outliers /
                rows
            ) * 100

            score -= min(
                outlier_percentage,
                20
            )

        score = max(0, score)

        st.metric(
            "Overall Dataset Health",
            f"{score:.1f}/100"
        )

        if score >= 80:

            st.success(
                "🟢 Dataset health looks good."
            )

        elif score >= 60:

            st.warning(
                "🟡 Dataset needs some cleaning."
            )

        else:

            st.error(
                "🔴 Dataset requires significant cleaning."
            )

    except Exception as e:

        st.error(
            f"Unable to read the dataset: {e}"
        )
