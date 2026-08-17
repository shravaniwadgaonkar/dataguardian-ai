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
                    "Lower Bound": round(lower, 2),
                    "Upper Bound": round(upper, 2)
                })

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
            else:
                st.success(
                    "✅ No potential outliers detected."
                )

        else:
            st.info("No numerical columns available.")

        # =========================
        # PRIVACY AUDIT
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

        else:

            privacy_risk = "LOW"

            st.success(
                "✅ No potentially sensitive column names detected."
            )

        st.caption(
            "Privacy detection is based on column names and "
            "is a heuristic, not a guarantee."
        )

        # =========================
        # BIAS AUDIT
        # =========================

        st.subheader("⚖️ Bias Audit")

        st.write(
            "Select a group column and an outcome column "
            "to compare outcome rates between groups."
        )

        categorical_columns = list(
            df.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        all_columns = list(df.columns)

        if len(categorical_columns) > 0:

            group_column = st.selectbox(
                "Group / Protected Attribute",
                categorical_columns
            )

            target_column = st.selectbox(
                "Outcome / Target",
                all_columns
            )

            if group_column and target_column:

                unique_targets = df[target_column].dropna().unique()

                if len(unique_targets) == 2:

                    positive_value = st.selectbox(
                        "Select the positive outcome",
                        unique_targets
                    )

                    temp = df[
                        [group_column, target_column]
                    ].dropna()

                    temp["positive"] = (
                        temp[target_column] ==
                        positive_value
                    )

                    bias_table = (
                        temp.groupby(group_column)["positive"]
                        .agg(["mean", "count"])
                        .reset_index()
                    )

                    bias_table["Outcome Rate (%)"] = (
                        bias_table["mean"] * 100
                    ).round(2)

                    bias_table = bias_table.drop(
                        columns=["mean"]
                    )

                    st.dataframe(
                        bias_table,
                        use_container_width=True
                    )

                    rates = bias_table[
                        "Outcome Rate (%)"
                    ]

                    if len(rates) >= 2:

                        disparity = (
                            rates.max() -
                            rates.min()
                        )

                        st.metric(
                            "Maximum Outcome Rate Difference",
                            f"{disparity:.2f}%"
                        )

                        if disparity >= 20:

                            st.error(
                                "🔴 Large outcome disparity detected."
                            )

                        elif disparity >= 10:

                            st.warning(
                                "🟡 Potential outcome disparity detected."
                            )

                        else:

                            st.success(
                                "🟢 No large outcome disparity detected."
                            )

                        st.caption(
                            "A disparity does not prove discrimination. "
                            "It indicates that further investigation may "
                            "be appropriate."
                        )

                else:

                    st.info(
                        "Bias analysis currently supports "
                        "binary outcomes only."
                    )

        else:

            st.info(
                "No categorical columns were found, so "
                "bias analysis cannot be performed automatically."
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
        # HEALTH SCORE
        # =========================

        st.subheader("🏥 Dataset Health Score")

        score = 100.0

        if rows > 0 and columns > 0:

            missing_percentage = (
                total_missing /
                (rows * columns)
            ) * 100

            score -= min(
                missing_percentage * 2,
                30
            )

        if rows > 0:

            duplicate_percentage = (
                duplicates / rows
            ) * 100

            score -= min(
                duplicate_percentage,
                20
            )

            outlier_percentage = (
                total_outliers / rows
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
            st.success("🟢 Dataset health looks good.")
        elif score >= 60:
            st.warning("🟡 Dataset needs some cleaning.")
        else:
            st.error(
                "🔴 Dataset requires significant cleaning."
            )

        # =========================
        # RECOMMENDATIONS
        # =========================

        st.subheader("💡 Recommendations")

        recommendations = []

        if total_missing > 0:
            recommendations.append(
                "Investigate and handle missing values."
            )

        if duplicates > 0:
            recommendations.append(
                "Review and remove duplicate records if appropriate."
            )

        if total_outliers > 0:
            recommendations.append(
                "Investigate potential outliers before machine learning."
            )

        if sensitive_columns:
            recommendations.append(
                "Consider masking or anonymizing potentially sensitive data."
            )

        if not recommendations:
            recommendations.append(
                "Dataset quality checks look good. Continue with exploratory analysis."
            )

        for recommendation in recommendations:
            st.write(f"• {recommendation}")

    except Exception as e:

        st.error(
            f"Unable to read the dataset: {e}"
        )
