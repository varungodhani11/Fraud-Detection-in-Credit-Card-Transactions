import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from prediction import (
    predict_fraud_batch,
    features,
    threshold
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("💳 Fraud Detection")

    st.write(
        "Machine learning application for identifying "
        "potentially fraudulent credit card transactions."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** XGBoost")
    st.write(f"**Classification Threshold:** {threshold:.4f}")
    st.write(f"**Input Features:** {len(features)}")

    st.divider()

    st.subheader("How It Works")

    st.write(
        """
        1. Upload transaction data
        2. Validate required features
        3. Apply the trained scaler
        4. Generate fraud probabilities
        5. Apply the classification threshold
        6. Review and download predictions
        """
    )

    st.divider()

    st.caption(
        "This application is intended for educational "
        "and analytical purposes."
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("Credit Card Fraud Detection")

st.write(
    "Upload transaction data to identify potentially "
    "fraudulent transactions using a trained XGBoost model."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the required transaction features.",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.success(
            "CSV file uploaded successfully."
        )

        # ----------------------------------------------------
        # DATASET PREVIEW
        # ----------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            uploaded_data.head(),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        st.subheader("Dataset Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:

            st.metric(
                "Rows",
                f"{uploaded_data.shape[0]:,}"
            )

        with info_col2:

            st.metric(
                "Columns",
                f"{uploaded_data.shape[1]:,}"
            )

        with info_col3:

            st.metric(
                "Required Features",
                f"{len(features)}"
            )

        # ----------------------------------------------------
        # FEATURE VALIDATION
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in features
            if feature not in uploaded_data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded CSV is missing required "
                "model features."
            )

            st.write("Missing features:")

            st.code(
                ", ".join(missing_features)
            )

        else:

            st.success(
                "All required model features are available."
            )

            st.divider()

            # ------------------------------------------------
            # FRAUD DETECTION
            # ------------------------------------------------

            st.subheader("🔍 Fraud Detection")

            if st.button(
                "Run Fraud Detection",
                type="primary",
                use_container_width=True
            ):

                prediction_data = uploaded_data[
                    features
                ].copy()

                # --------------------------------------------
                # BATCH PREDICTION
                # --------------------------------------------

                with st.spinner(
                    "Running fraud detection..."
                ):

                    predictions, fraud_probabilities = (
                        predict_fraud_batch(
                            prediction_data
                        )
                    )

                # --------------------------------------------
                # CREATE RESULTS
                # --------------------------------------------

                results = uploaded_data.copy()

                results["Prediction"] = predictions

                results["Fraud Probability"] = (
                    fraud_probabilities
                )

                # --------------------------------------------
                # SUMMARY VALUES
                # --------------------------------------------

                total_transactions = len(results)

                fraud_count = (
                    results["Prediction"] == "Fraud"
                ).sum()

                normal_count = (
                    results["Prediction"] == "Normal"
                ).sum()

                fraud_percentage = (
                    fraud_count / total_transactions * 100
                    if total_transactions > 0
                    else 0
                )

                # --------------------------------------------
                # RESULTS SUMMARY
                # --------------------------------------------

                st.divider()

                st.subheader(
                    "📊 Detection Summary"
                )

                result_col1, result_col2, result_col3, result_col4 = (
                    st.columns(4)
                )

                with result_col1:

                    st.metric(
                        "Total Transactions",
                        f"{total_transactions:,}"
                    )

                with result_col2:

                    st.metric(
                        "Potential Fraud",
                        f"{fraud_count:,}"
                    )

                with result_col3:

                    st.metric(
                        "Normal Transactions",
                        f"{normal_count:,}"
                    )

                with result_col4:

                    st.metric(
                        "Fraud Rate",
                        f"{fraud_percentage:.2f}%"
                    )

                # --------------------------------------------
                # FRAUD ALERT
                # --------------------------------------------

                if fraud_count > 0:

                    st.warning(
                        f"⚠️ {fraud_count:,} transaction(s) "
                        "were flagged as potentially fraudulent."
                    )

                else:

                    st.success(
                        "✅ No transactions were flagged "
                        "as potentially fraudulent."
                    )

                # --------------------------------------------
                # PREDICTION RESULTS
                # --------------------------------------------

                st.subheader(
                    "📋 Prediction Results"
                )

                display_results = results.copy()

                display_results[
                    "Fraud Probability"
                ] = (
                    display_results[
                        "Fraud Probability"
                    ].round(4)
                )

                st.dataframe(
                    display_results,
                    use_container_width=True
                )

                # --------------------------------------------
                # DOWNLOAD RESULTS
                # --------------------------------------------

                results_csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=results_csv,
                    file_name=(
                        "fraud_detection_predictions.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

                # --------------------------------------------
                # MODEL INFORMATION
                # --------------------------------------------

                st.divider()

                st.subheader(
                    "⚙️ Prediction Configuration"
                )

                config_col1, config_col2 = st.columns(2)

                with config_col1:

                    st.metric(
                        "Model",
                        "XGBoost"
                    )

                with config_col2:

                    st.metric(
                        "Fraud Threshold",
                        f"{threshold:.4f}"
                    )

                st.caption(
                    "Transactions with a predicted fraud "
                    "probability at or above the selected "
                    "threshold are classified as Fraud."
                )

    except Exception as error:

        st.error(
            "Unable to process the uploaded CSV file."
        )

        st.exception(error)

else:

    # ========================================================
    # INITIAL APPLICATION STATE
    # ========================================================

    st.info(
        "Upload a CSV file above to start fraud detection."
    )