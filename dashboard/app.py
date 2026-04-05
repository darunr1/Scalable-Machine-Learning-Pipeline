"""
Streamlit Dashboard for the ML Pipeline.

Displays:
    - Model performance over time
    - Drift metrics
    - Feature distributions
    - Recent predictions
    - Model registry
"""

import json
import os
import sys
import glob

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_path


def load_model_registry():
    """Load model registry metadata."""
    config = load_config()
    metadata_file = resolve_path(config["registry"]["metadata_file"])
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {"models": [], "production_version": None}


def load_prediction_logs(limit=500):
    """Load recent prediction logs."""
    config = load_config()
    log_file = resolve_path(config["monitoring"]["prediction_log"])
    if not os.path.exists(log_file):
        return []
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs[-limit:] if limit else logs


def load_drift_reports():
    """Load drift detection reports."""
    config = load_config()
    reports_dir = resolve_path(config["drift"]["reports_dir"])
    if not os.path.exists(reports_dir):
        return []
    reports = []
    for f in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        with open(f, "r") as fp:
            reports.append(json.load(fp))
    return reports


def load_monitoring_reports():
    """Load monitoring reports."""
    config = load_config()
    reports_dir = resolve_path(config["monitoring"]["reports_dir"])
    if not os.path.exists(reports_dir):
        return []
    reports = []
    for f in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        with open(f, "r") as fp:
            reports.append(json.load(fp))
    return reports


# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 ML Pipeline Dashboard")
st.markdown("Production monitoring for the weather prediction pipeline")

# ── Sidebar ─────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "🔍 Drift Metrics", "📈 Feature Distributions",
     "🎯 Predictions", "📦 Model Registry"],
)

# ════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ════════════════════════════════════════════════════════════════════════
if page == "📊 Model Performance":
    st.header("📊 Model Performance Over Time")

    registry = load_model_registry()
    models = registry.get("models", [])

    if not models:
        st.info("No models registered yet. Run the training pipeline first.")
    else:
        # Metrics comparison chart
        versions = [m["version"] for m in models]
        mae_values = [m["metrics"].get("mae", 0) for m in models]
        rmse_values = [m["metrics"].get("rmse", 0) for m in models]
        r2_values = [m["metrics"].get("r2", 0) for m in models]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Best MAE",
                f"{min(mae_values):.4f}",
                f"v{versions[mae_values.index(min(mae_values))]}",
            )
        with col2:
            st.metric(
                "Best RMSE",
                f"{min(rmse_values):.4f}",
                f"v{versions[rmse_values.index(min(rmse_values))]}",
            )
        with col3:
            st.metric(
                "Best R²",
                f"{max(r2_values):.4f}",
                f"v{versions[r2_values.index(max(r2_values))]}",
            )

        st.subheader("MAE & RMSE by Model Version")
        chart_data = pd.DataFrame({
            "Version": [f"v{v}" for v in versions],
            "MAE": mae_values,
            "RMSE": rmse_values,
        }).set_index("Version")
        st.bar_chart(chart_data)

        st.subheader("R² Score by Model Version")
        r2_data = pd.DataFrame({
            "Version": [f"v{v}" for v in versions],
            "R²": r2_values,
        }).set_index("Version")
        st.line_chart(r2_data)

        # Production model info
        prod_version = registry.get("production_version")
        if prod_version:
            st.success(f"🟢 Production model: **v{prod_version}**")

# ════════════════════════════════════════════════════════════════════════
# PAGE: Drift Metrics
# ════════════════════════════════════════════════════════════════════════
elif page == "🔍 Drift Metrics":
    st.header("🔍 Drift Detection Metrics")

    reports = load_drift_reports()

    if not reports:
        st.info("No drift reports available. Run a drift check first.")
    else:
        latest = reports[-1]

        # Alert status
        if latest.get("drift_detected"):
            st.error("⚠️ DATA DRIFT DETECTED")
            for alert in latest.get("alerts", []):
                st.warning(alert)
        else:
            st.success("✅ No drift detected")

        # Per-feature KS statistics
        st.subheader("KS Statistics by Feature")
        features = latest.get("features", {})
        if features:
            feature_names = list(features.keys())
            ks_stats = [features[f]["ks_statistic"] for f in feature_names]
            p_values = [features[f]["p_value"] for f in feature_names]

            drift_df = pd.DataFrame({
                "Feature": feature_names,
                "KS Statistic": ks_stats,
                "P-Value": p_values,
                "Drift": ["🔴 Yes" if features[f]["drift_detected"] else "🟢 No" for f in feature_names],
            })
            st.dataframe(drift_df, use_container_width=True)

            # KS statistic bar chart
            chart_df = pd.DataFrame({
                "Feature": feature_names,
                "KS Statistic": ks_stats,
            }).set_index("Feature")
            st.bar_chart(chart_df)

        # Historical drift reports
        if len(reports) > 1:
            st.subheader("Drift History")
            drift_history = []
            for r in reports:
                drift_history.append({
                    "Timestamp": r.get("timestamp", ""),
                    "Drift Detected": "Yes" if r.get("drift_detected") else "No",
                    "Alerts": len(r.get("alerts", [])),
                })
            st.dataframe(pd.DataFrame(drift_history), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE: Feature Distributions
# ════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Distributions":
    st.header("📈 Feature Distributions")

    logs = load_prediction_logs()

    if not logs:
        st.info("No prediction logs available.")
    else:
        # Extract features from logs
        feature_keys = list(logs[0].get("input_features", {}).keys())

        if feature_keys:
            selected_feature = st.selectbox("Select Feature", feature_keys)

            values = [
                log["input_features"][selected_feature]
                for log in logs
                if selected_feature in log.get("input_features", {})
                and isinstance(log["input_features"][selected_feature], (int, float))
            ]

            if values:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{np.mean(values):.2f}")
                with col2:
                    st.metric("Std", f"{np.std(values):.2f}")
                with col3:
                    st.metric("Min", f"{np.min(values):.2f}")
                with col4:
                    st.metric("Max", f"{np.max(values):.2f}")

                # Histogram
                st.subheader(f"Distribution of {selected_feature}")
                hist_df = pd.DataFrame({selected_feature: values})
                st.bar_chart(hist_df[selected_feature].value_counts().sort_index())
        else:
            st.info("No features found in prediction logs.")

# ════════════════════════════════════════════════════════════════════════
# PAGE: Predictions
# ════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predictions":
    st.header("🎯 Recent Predictions")

    logs = load_prediction_logs(limit=100)

    if not logs:
        st.info("No predictions yet. Make predictions via the API.")
    else:
        # Summary metrics
        predictions = [log["prediction"] for log in logs]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(logs))
        with col2:
            st.metric("Mean Prediction", f"{np.mean(predictions):.2f}°C")
        with col3:
            st.metric("Min", f"{np.min(predictions):.2f}°C")
        with col4:
            st.metric("Max", f"{np.max(predictions):.2f}°C")

        # Prediction timeline
        st.subheader("Prediction Timeline")
        pred_df = pd.DataFrame({
            "Timestamp": [log["timestamp"] for log in logs],
            "Prediction (°C)": predictions,
        })
        pred_df["Timestamp"] = pd.to_datetime(pred_df["Timestamp"])
        pred_df = pred_df.set_index("Timestamp")
        st.line_chart(pred_df)

        # Recent predictions table
        st.subheader("Recent Predictions")
        table_data = []
        for log in reversed(logs[-20:]):
            table_data.append({
                "Time": log["timestamp"][:19],
                "Prediction (°C)": round(log["prediction"], 2),
                "Model": f"v{log.get('model_version', '?')}",
                "Confidence σ": log.get("confidence", {}).get("std", "N/A"),
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE: Model Registry
# ════════════════════════════════════════════════════════════════════════
elif page == "📦 Model Registry":
    st.header("📦 Model Registry")

    registry = load_model_registry()
    models = registry.get("models", [])

    if not models:
        st.info("No models registered yet.")
    else:
        prod_version = registry.get("production_version")
        st.success(f"🟢 Production model: **v{prod_version}**")

        # Model table
        table_data = []
        for m in models:
            table_data.append({
                "Version": f"v{m['version']}",
                "MAE": m["metrics"].get("mae", "N/A"),
                "RMSE": m["metrics"].get("rmse", "N/A"),
                "R²": m["metrics"].get("r2", "N/A"),
                "Registered": m.get("registered_at", "")[:19],
                "Production": "✅" if m.get("is_production") else "",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Model details expander
        for m in models:
            with st.expander(f"Model v{m['version']} Details"):
                st.json(m)

# ── Footer ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**ML Pipeline Dashboard** v1.0")
st.sidebar.markdown("Built with Streamlit")
