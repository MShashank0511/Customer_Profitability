import numpy as np
import os
import json
import shutil
import streamlit as st
from xgboost import XGBClassifier
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
app_version = st.session_state.get("app_version", "Customized")
is_mvp = app_version == "MVP"


# --- Constants and Configuration ---
DEFAULT_DATA_DIR = "default_data"
DATA_REGISTRY_BASE_DIR = "data_registry"

MODEL_INPUT_SOURCES = {
    "Charge-Off Model (for COF)": {
        "session_state_key_suffix": "Charge-Off Model",
        "target_column_primary": "COF_EVENT_LABEL",
        "data_registry_subfolder_actual": "Charge-Off_Model"
    },
    "Prepayment Model (for Prepayment)": {
        "session_state_key_suffix": "Prepayment Model",
        "target_column_primary": "PREPAYMENT_EVENT_LABEL",
        "data_registry_subfolder_actual": "Prepayment_Model"
    },
    "Forecast Model (for Profitability)": {
        "session_state_key_suffix": "Forecast_Model",
        "target_column_primary": "Profitability_GBP",
        "data_registry_subfolder_actual": "Forecast Model"
    }
}
def load_default_data():
    default_files = {
        "COF_EVENT_LABEL": os.path.join(DEFAULT_DATA_DIR, "default_COF_EVENT_LABEL_test_data.parquet"),
        "PREPAYMENT_EVENT_LABEL": os.path.join(DEFAULT_DATA_DIR, "default_PREPAYMENT_EVENT_LABEL_test_data.parquet"),
        "Profitability_GBP": os.path.join(DEFAULT_DATA_DIR, "default_Profitability_GBP_test_data.parquet"),
        "on_us_data": os.path.join(DEFAULT_DATA_DIR, "default_on_us_data.parquet"),
    }
    for key, file_path in default_files.items():
        if key not in st.session_state or not isinstance(st.session_state.get(key), pd.DataFrame):
            if os.path.exists(file_path):
                try:
                    st.session_state[key] = pd.read_parquet(file_path)
                except Exception as e:
                    st.warning(f"⚠️ Default file for {key} at {file_path} could not be loaded: {e}")
            elif key == "on_us_data":
                 st.warning(f"⚠️ Default main data file ('on_us_data.parquet') for '{key}' not found at {file_path}.")

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown")
    df = df.clip(lower=-1e9, upper=1e9)
    return df

st.title("🔧 Model Development")
st.markdown("""
<div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f9f9f9; color: black;">
    <h3 style="color: #4CAF50;">📊 Steps to Follow on This Page</h3>
    <ol>
        <li><b>Select Model Type</b><br>
            Choose the appropriate model type for your selected dataset.</li>
        <li><b>Set Train-Test Split</b><br>
            Specify the train-test split ratio to define how your data will be divided for training and evaluation.</li>
        <li><b>Optimize Hyperparameters</b><br>
            Identify the best hyperparameters for your model based on the desired evaluation metric. From the suggested options, select the one that best fits your needs.</li>
        <li><b>Repeat for All Models</b><br>
            Once you've completed these steps for one model, repeat the same process for the remaining models.</li>
        <li><b>Proceed to Performance Monitoring Page</b><br>
            After configuring all three models, continue to the next page to move forward with the project workflow.</li>
    </ol>
    
</div>
""", unsafe_allow_html=True)
if "confirmed_model_outputs" not in st.session_state:
    st.session_state["confirmed_model_outputs"] = {}

if "model_development_state" not in st.session_state:
    st.session_state["model_development_state"] = {}

modeling_tasks = list(MODEL_INPUT_SOURCES.keys())

# --- Track which model sections to show ---
if "model_dev_visible_sections" not in st.session_state:
    st.session_state["model_dev_visible_sections"] = {modeling_tasks[0]: True}
for m in modeling_tasks[1:]:
    if m not in st.session_state["model_dev_visible_sections"]:
        st.session_state["model_dev_visible_sections"][m] = False

for model_index, model_name in enumerate(modeling_tasks):
    # Only show this section if visible
    if not st.session_state["model_dev_visible_sections"].get(model_name, False):
        continue

    model_config = MODEL_INPUT_SOURCES[model_name]
    active_model_suffix_for_path = model_config["session_state_key_suffix"]
    target_column = model_config["target_column_primary"]
    data_registry_subfolder = model_config["data_registry_subfolder_actual"]

    st.markdown("---")
    st.subheader(f"📊 Building {model_name}")

    # --- Data Loading Logic ---
    df_full = None
    data_source_message = ""
    data_path_key_from_prev_page = f"{active_model_suffix_for_path}_final_dataset_path"
    data_path_from_session = st.session_state.get(data_path_key_from_prev_page)

    if data_path_from_session and os.path.exists(data_path_from_session):
        try:
            df_full = pd.read_parquet(data_path_from_session)
        except Exception as e:
            st.error(f"Error loading data from session state path '{data_path_from_session}': {e}. Attempting fallbacks.")
            df_full = None

    if df_full is None:
        constructed_path = os.path.join(
            DATA_REGISTRY_BASE_DIR,
            data_registry_subfolder,
            f"{target_column}_final_dataset.parquet"
        )
        if os.path.exists(constructed_path):
            try:
                df_full = pd.read_parquet(constructed_path)
            except Exception as e:
                st.error(f"Error loading data from constructed path '{constructed_path}': {e}. Attempting general default data load.")
                df_full = None
        else:
            st.error(f"Constructed data path '{constructed_path}' not found. Please verify the subfolder name and file name.")

    if df_full is None:
        st.warning(f"Could not load specific data for '{model_name}' from data_registry. Falling back to the general default dataset ('on_us_data.parquet').")
        if "on_us_data" not in st.session_state or not isinstance(st.session_state.get("on_us_data"), pd.DataFrame):
            load_default_data()
        if "on_us_data" in st.session_state and isinstance(st.session_state["on_us_data"], pd.DataFrame):
            df_full = st.session_state["on_us_data"].copy()
            data_source_message = "Data loaded from general default: `default_data/default_on_us_data.parquet` (Fallback)."
            st.info(data_source_message)
        else:
            st.error("CRITICAL: Failed to load any data. Please check your data files and paths.")
            st.stop()

    if 'timestamp' in df_full.columns:
        df_full = df_full.drop(columns=['timestamp'])

    target_variables_list = ["Profitability_GBP", "COF_EVENT_LABEL", "PREPAYMENT_EVENT_LABEL"]
    if target_column not in df_full.columns:
        st.error(f"FATAL ERROR: The target variable '{target_column}' (for selected task '{model_name}') is NOT PRESENT in the currently loaded dataset. Please ensure the data source is correct or select a different model task.")
        st.stop()

    # --- Data Overview ---
    st.subheader(f"📊 Data Overview for : {model_name}")
    
    st.write(f"DataFrame shape: {df_full.shape}")
    if st.checkbox(f"Show overview of the loaded dataset? ({model_name})", key=f"show_df_full_head_model_dev_{model_index}"):
        st.dataframe(df_full.head())
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # --- Model Selection (Moved here, after Data Overview, before Sub-sampling) ---
    st.subheader(f"📚 Step 1 : Select AI Model Type for : {model_name}")
    
    # Infer target type
    if df_full[target_column].nunique() < 10 and df_full[target_column].dtype in ['int64', 'float64', 'object', 'category', 'bool']:
        target_type = "Classification"
        if df_full[target_column].dtype == 'object' or df_full[target_column].dtype == 'category' or df_full[target_column].dtype == 'bool':
            try:
                le = LabelEncoder()
                df_full[target_column] = le.fit_transform(df_full[target_column])
                st.info(f"Encoded categorical/boolean target '{target_column}' using LabelEncoder.")
                try:
                    mapping_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                    st.write(f"Mapping: {mapping_dict}")
                except Exception as map_e:
                    st.write(f"Could not display exact mapping: {map_e}")
                if len(le.classes_) > 2:
                    st.warning(f"Multi-class classification target '{target_column}' detected ({len(le.classes_)} classes). Some metrics and SHAP plots may behave differently.")
            except Exception as e:
                st.error(f"Could not encode target variable '{target_column}': {e}")
                st.stop()
    elif not pd.api.types.is_numeric_dtype(df_full[target_column]):
        st.warning(f"Target '{target_column}' is non-numeric but inferred as Classification. Attempting LabelEncoding.")
        try:
            le = LabelEncoder()
            df_full[target_column] = le.fit_transform(df_full[target_column])
        except Exception as e:
            st.error(f"Could not encode non-numeric target '{target_column}': {e}")
            st.stop()
        target_type = "Classification"
    else:
        target_type = "Regression"
        if not pd.api.types.is_numeric_dtype(df_full[target_column]):
            st.error(f"Target '{target_column}' inferred as Regression but is not numeric ({df_full[target_column].dtype}). Please check data.")
            st.stop()

    st.markdown(f"**Target Variable for current task:** {target_column}")
    st.markdown(f"**Problem Type:** {target_type}")

    # Model selection UI
    if target_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression,
            "LGBM Classifier": LGBMClassifier,
            "Random Forest Classifier": RandomForestClassifier,
            "XGBoost Classifier": XGBClassifier
        }
    else:
        models = {"Linear Regression": LinearRegression}

    default_model_name = list(models.keys())[0]
    if "selected_model" in model_config and model_config["selected_model"] in models:
        default_model_name = model_config["selected_model"]

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        index=list(models.keys()).index(default_model_name),
        key=f"model_{model_name.replace(' ', '_')}"
    )
    model_config["selected_model"] = selected_model
    model_class = models[selected_model]
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # --- Sub-sampling (Collapsible Section) ---
    with st.expander("🔍 Sub-sampling (optional)", expanded=False):
        sample_frac = st.slider(
            "Select sub-sample fraction for the current task", 0.01, 1.0,
            value=1.0, key=f"sample_frac_{active_model_suffix_for_path}"
        )
        if sample_frac < 1.0:
            df_full = df_full.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            st.info(f"Using a {sample_frac:.2f} fraction of the data ({len(df_full)} rows) for '{model_name}'.")

    # --- Target Variable Analysis (Collapsible Section) ---
    with st.expander(f"🎯 Target Variable Analysis for: {target_column}", expanded=False):
        if target_type == "Regression":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_full[target_column].dropna(), kde=True, ax=ax, color="blue")
            ax.set_title(f"Distribution of {target_column}")
            ax.set_xlabel(target_column)
            st.pyplot(fig)
            plt.close(fig)
        else:
            try:
                unique_targets_after_encoding = df_full[target_column].unique()
                if len(unique_targets_after_encoding) == 2:
                    positive_class_label = 1
                    if positive_class_label in df_full[target_column].value_counts(normalize=True):
                        event_rate_value = df_full[target_column].value_counts(normalize=True).get(positive_class_label, 0) * 100
                        st.metric(
                            label="Event Rate (%)",
                            value=f"{event_rate_value:.2f}%",
                            help=f"Percentage of positive events (class {positive_class_label}) in the target variable '{target_column}'."
                        )
                    else:
                        st.warning(f"Positive class label '{positive_class_label}' not found for event rate calculation in '{target_column}'.")
                else:
                    st.info(f"Target '{target_column}' has {len(unique_targets_after_encoding)} unique encoded values. Displaying frequency.")

                frequency_df = df_full[target_column].value_counts().reset_index()
                frequency_df.columns = [target_column, "count"]
                if not pd.api.types.is_numeric_dtype(frequency_df[target_column]):
                    frequency_df[target_column] = frequency_df[target_column].astype(str)
                fig = px.bar(
                    frequency_df,
                    x=target_column,
                    y="count",
                    text="count",
                    labels={target_column: str(target_column), "count": "Frequency"},
                    title=f"Frequency of Encoded {target_column}"
                )
                fig.update_traces(textposition="inside")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying target distribution for {target_column}: {e}")
                st.exception(e)
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # --- Feature Selection & Train/Test Split ---
    st.subheader("📊 Step 2 :View Features & Split Data")
    timestamp_cols = ["Timestamp_x", "Timestamp_y", "TERM_OF_LOAN_y","Timestamp","Application_ID_x", "Application_ID_y", "Application_ID"]
    excluded_cols_from_features = target_variables_list + timestamp_cols
    selectable_features = [col for col in df_full.columns if col not in excluded_cols_from_features and col != target_column and col != "Application_ID"]
    selected_features_default = model_config.get("feature_columns", selectable_features)
    selected_features_default = [f for f in selected_features_default if f in selectable_features]
    feature_columns = selectable_features
    model_config["feature_columns"] = feature_columns

    if not feature_columns:
        st.error("No valid features available in the dataset.")
        st.stop()

    with st.expander("📂 View All Features and Their Data Types", expanded=False):
        st.markdown("### Features Overview")
        features_df = pd.DataFrame({
            "Feature Name": feature_columns,
            "Data Type": [df_full[col].dtype for col in feature_columns]
        })
        st.dataframe(features_df, use_container_width=True)

    df_full[feature_columns] = preprocess_dataset(df_full[feature_columns])
    X = df_full[feature_columns]
    y = df_full[target_column]
    original_indices = df_full.index

    default_test_size = model_config.get("test_size", 0.2)
    test_size = st.slider(
        "Select test size for splitting the data", 0.01, 0.5,
        default_test_size, key=f"test_size_{active_model_suffix_for_path}"
    )
    model_config["test_size"] = test_size

    if len(X) != len(y) or len(X) != len(original_indices):
        st.error(f"Inconsistent lengths after dropping NaN values: X ({len(X)}), y ({len(y)}), original_indices ({len(original_indices)}). Please check the data.")
        st.stop()

    stratify_y = None
    if target_type == "Classification":
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) > 1 and np.min(counts) >= 2:
            stratify_y = y
        elif len(unique_classes) > 1:
            st.warning(f"Stratification skipped for target '{target_column}': Some classes have fewer than 2 samples. Minimum counts per class: {np.min(counts)}.")

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, original_indices, test_size=test_size, random_state=42, stratify=stratify_y)

    st.info(f"Train set size: {len(X_train)} rows | Test set size: {len(X_test)} rows for '{model_name}'.")
    
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    # --- Hyperparameter Iterations & Model Training ---
    st.subheader("🏆 Step 3 :Hyperparameter Iterations")

    task_key = f"{model_name.replace(' ', '_')}_task"
    selected_model_task_name = model_name  # <-- Define this before use

    if is_mvp:
        st.info("This section is disabled in MVP mode. Only one run will be performed for the selected model with default hyperparameters.")
        # Prepare default parameters for the selected model
        param_dist = {
            "Logistic Regression": {"C": [1.0], "solver": ["lbfgs"], "max_iter": [100]},
            "LGBM Classifier": {"n_estimators": [100], "max_depth": [-1], "learning_rate": [0.1]},
            "Random Forest Classifier": {"n_estimators": [100], "max_depth": [None]},
            "XGBoost Classifier": {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], 'use_label_encoder': [False], 'eval_metric': ['logloss']},
            "Linear Regression": {"fit_intercept": [True]},
        }
        sampled_params = [ {k: v[0] for k, v in param_dist[selected_model].items()} ]

        # No metric selection in MVP mode, just run and show results
        if st.button(f"Run Model for {selected_model_task_name}", key=f"run_{task_key}_mvp"):
            iteration_results_list = []
            try:
                params_iter = sampled_params[0]
                if selected_model == "XGBoost Classifier" and 'use_label_encoder' in params_iter:
                    params_iter_copy = params_iter.copy()
                    params_iter_copy['use_label_encoder'] = False
                    model_instance = model_class(**params_iter_copy)
                else:
                    model_instance = model_class(**params_iter)
                model_instance.fit(X_train, y_train)
                y_pred_test = model_instance.predict(X_test)
                metrics_dict = {"Parameters": params_iter}
                if target_type == "Classification":
                    metrics_dict["Accuracy"] = accuracy_score(y_test, y_pred_test)
                    metrics_dict["Precision"] = precision_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                    metrics_dict["Recall"] = recall_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                    metrics_dict["F1 Score"] = f1_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                    if hasattr(model_instance, "predict_proba"):
                        y_proba_test = model_instance.predict_proba(X_test)[:, 1]
                        metrics_dict["ROC-AUC"] = roc_auc_score(y_test, y_proba_test)
                    else:
                        metrics_dict["ROC-AUC"] = "N/A"
                    metrics_dict["Confusion Matrix"] = confusion_matrix(y_test, y_pred_test).tolist()
                else:
                    metrics_dict["Mean Squared Error (MSE)"] = mean_squared_error(y_test, y_pred_test)
                    metrics_dict["Root Mean Squared Error (RMSE)"] = np.sqrt(metrics_dict["Mean Squared Error (MSE)"])
                    metrics_dict["Mean Absolute Percentage Error (MAPE)"] = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-9))) * 100 if len(y_test) > 0 else 0
                iteration_results_list.append({
                    "iteration_num": 1,
                    "params": params_iter,
                    "metrics": metrics_dict,
                    "model_object": model_instance,
                    "test_indices": test_indices.tolist()
                })
            except Exception as e_iter:
                st.warning(f"⚠️ Model run failed: {e_iter}")
                iteration_results_list.append({
                    "iteration_num": 1,
                    "params": sampled_params[0],
                    "metrics": {"Error": str(e_iter)},
                    "model_object": None,
                    "test_indices": None
                })
            st.session_state[f"iteration_results_{task_key}"] = iteration_results_list

        # --- Display Single Result as "Results for Selected Model" ---
        iteration_results_for_task_display = st.session_state.get(f"iteration_results_{task_key}", [])
        if iteration_results_for_task_display:
            st.subheader(f"📊 Results for Selected Model ({selected_model})")
            iter_data = iteration_results_for_task_display[0]
            with st.expander("Show Results", expanded=True):
                st.markdown("##### 🔧 Hyperparameters")
                st.json(iter_data["params"])
                st.markdown("##### 📊 Metrics")
                if "Error" in iter_data["metrics"]:
                    st.error(f"This run failed: {iter_data['metrics']['Error']}")
                else:
                    display_metrics_data = {k: v for k, v in iter_data["metrics"].items() if k not in ["Confusion Matrix", "Parameters"]}
                    num_metrics_cols_disp = min(len(display_metrics_data), 3)
                    if num_metrics_cols_disp > 0:
                        metric_cols_disp = st.columns(num_metrics_cols_disp)
                        col_idx_disp = 0
                        for metric_name_disp, value_disp in display_metrics_data.items():
                            current_col_disp = metric_cols_disp[col_idx_disp % num_metrics_cols_disp]
                            if isinstance(value_disp, (int, float, np.float_)):
                                current_col_disp.metric(metric_name_disp, f"{value_disp:.4f}")
                            else:
                                current_col_disp.metric(metric_name_disp, str(value_disp))
                            col_idx_disp += 1
                    if "Confusion Matrix" in iter_data["metrics"] and target_type == "Classification":
                        st.markdown("###### Confusion Matrix")
                        cm_data_plot = np.array(iter_data["metrics"]["Confusion Matrix"])
                        fig_cm_plot, ax_cm_plot = plt.subplots(figsize=(4,3))
                        sns.heatmap(cm_data_plot, annot=True, fmt="d", cmap="Blues", ax=ax_cm_plot, cbar=False)
                        ax_cm_plot.set_xlabel("Predicted")
                        ax_cm_plot.set_ylabel("Actual")
                        st.pyplot(fig_cm_plot)
                        plt.close(fig_cm_plot)
                    if iter_data["model_object"] and X_test.shape[0] > 0 and X_train.shape[0] > 0:
                        st.markdown("###### SHAP Summary Plot (on Test Set)")
                        try:
                            model_to_explain_shap = iter_data["model_object"]
                            if isinstance(model_to_explain_shap, (RandomForestClassifier, XGBClassifier, LGBMClassifier)) or \
                               (hasattr(model_to_explain_shap, '_estimator_type') and model_to_explain_shap._estimator_type == "regressor" and not isinstance(model_to_explain_shap, LinearRegression)):
                                explainer_shap = shap.Explainer(model_to_explain_shap, X_train)
                            else:
                                masker_shap = shap.maskers.Independent(X_train, max_samples=100)
                                explainer_shap = shap.Explainer(model_to_explain_shap, masker_shap)
                            shap_values_plot = explainer_shap(X_test)
                            fig_shap_plot, ax_shap_plot = plt.subplots()
                            if isinstance(shap_values_plot, list) and len(shap_values_plot) == 2:
                                shap.summary_plot(shap_values_plot[1], X_test, plot_type="dot", show=False, plot_size=None)
                            elif hasattr(shap_values_plot, "values"):
                                shap.summary_plot(shap_values_plot, X_test, plot_type="dot", show=False, plot_size=None)
                            else:
                                shap.summary_plot(shap_values_plot, X_test, plot_type="dot", show=False, plot_size=None)
                            st.pyplot(fig_shap_plot)
                            plt.close(fig_shap_plot)
                        except Exception as e_shap:
                            st.warning(f"⚠️ Could not generate SHAP summary plot: {e_shap}")

    else:
        # --- (copy your original multi-iteration code here) ---
        # --- Hyperparameter Iterations & Model Training ---
        
        param_dist = {
            "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "solver": ["liblinear", "lbfgs"], "max_iter": [100, 200, 300]},
            "LGBM Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7, -1], "learning_rate": [0.01, 0.05, 0.1]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7, 10, None]},
            "XGBoost Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1], 'use_label_encoder': [False], 'eval_metric': ['logloss', 'auc']},
            "Linear Regression": {"fit_intercept": [True, False]},
        }
        default_n_iterations = model_config.get("n_iterations", 5 if target_type == "Classification" else 1)
        task_key = f"{model_name.replace(' ', '_')}_task"
        n_iterations_to_run = st.number_input("Number of hyperparameter iterations to try", min_value=1, max_value=50, value=default_n_iterations, key=f"n_iter_{task_key}")
        model_config["n_iterations"] = n_iterations_to_run
        try:
            total_combinations = 1
            for p_values in param_dist[selected_model].values():
                total_combinations *= len(p_values)
            actual_n_iter = min(n_iterations_to_run, total_combinations)
            if actual_n_iter < n_iterations_to_run:
                st.info(f"Reducing iterations to {actual_n_iter} as it's the maximum number of unique parameter combinations for {selected_model}.")
            sampled_params = list(ParameterSampler(param_dist[selected_model], n_iter=actual_n_iter, random_state=42))
        except ValueError as e:
            st.error(f"Error sampling parameters: {e}. Check parameter distribution. Using default parameters.")
            sampled_params = [{}]

        if target_type == "Classification":
            metric_options = ["ROC-AUC", "F1 Score", "Accuracy", "Precision", "Recall"]
        else:
            metric_options = ["Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)", "Mean Absolute Percentage Error (MAPE)"]

        default_metric_eval = metric_options[0]
        if "selected_metric" in model_config and model_config["selected_metric"] in metric_options:
            default_metric_eval = model_config["selected_metric"]

        selected_metric_eval = st.selectbox(
            "Select Primary Metric to Evaluate Iterations By",
            metric_options,
            index=metric_options.index(default_metric_eval),
            key=f"metric_eval_{task_key}"
        )
        model_config["selected_metric"] = selected_metric_eval

        selected_model_task_name = model_name
        if st.button(f"Run Model Iterations for {selected_model_task_name}", key=f"run_{task_key}"):
            iteration_results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, params_iter in enumerate(sampled_params):
                status_text.text(f"Running iteration {i+1}/{len(sampled_params)} for {selected_model}...")
                progress_bar.progress((i + 1) / len(sampled_params))
                try:
                    if selected_model == "XGBoost Classifier" and 'use_label_encoder' in params_iter:
                        params_iter_copy = params_iter.copy()
                        params_iter_copy['use_label_encoder'] = False
                        model_instance = model_class(**params_iter_copy)
                    else:
                        model_instance = model_class(**params_iter)
                    model_instance.fit(X_train, y_train)
                    y_pred_test = model_instance.predict(X_test)
                    metrics_dict = {"Parameters": params_iter}
                    if target_type == "Classification":
                        metrics_dict["Accuracy"] = accuracy_score(y_test, y_pred_test)
                        metrics_dict["Precision"] = precision_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                        metrics_dict["Recall"] = recall_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                        metrics_dict["F1 Score"] = f1_score(y_test, y_pred_test, zero_division=0, pos_label=1)
                        if hasattr(model_instance, "predict_proba"):
                            y_proba_test = model_instance.predict_proba(X_test)[:, 1]
                            metrics_dict["ROC-AUC"] = roc_auc_score(y_test, y_proba_test)
                        else:
                            metrics_dict["ROC-AUC"] = "N/A"
                        metrics_dict["Confusion Matrix"] = confusion_matrix(y_test, y_pred_test).tolist()
                    else:
                        metrics_dict["Mean Squared Error (MSE)"] = mean_squared_error(y_test, y_pred_test)
                        metrics_dict["Root Mean Squared Error (RMSE)"] = np.sqrt(metrics_dict["Mean Squared Error (MSE)"])
                        metrics_dict["Mean Absolute Percentage Error (MAPE)"] = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-9))) * 100 if len(y_test) > 0 else 0
                    iteration_results_list.append({
                        "iteration_num": i + 1,
                        "params": params_iter,
                        "metrics": metrics_dict,
                        "model_object": model_instance,
                        "test_indices": test_indices.tolist()
                    })
                except Exception as e_iter:
                    st.warning(f"⚠️ Skipping iteration {i+1} for {selected_model} due to error: {e_iter}")
                    iteration_results_list.append({
                        "iteration_num": i + 1,
                        "params": params_iter,
                        "metrics": {"Error": str(e_iter)},
                        "model_object": None,
                        "test_indices": None
                    })
            st.session_state[f"iteration_results_{task_key}"] = iteration_results_list
            status_text.text("Model iterations complete.")
            progress_bar.empty()

        # --- Display Iteration Results ---
        iteration_results_for_task_display = st.session_state.get(f"iteration_results_{task_key}", [])
        if iteration_results_for_task_display:
            st.subheader(f"📊 Iteration Results (Model: {selected_model})")
            is_higher_better_metric = not any(m_eval in selected_metric_eval for m_eval in ["MSE", "RMSE", "MAPE"])
            valid_iterations_for_sorting = [
                res for res in iteration_results_for_task_display
                if "Error" not in res["metrics"] and isinstance(res["metrics"].get(selected_metric_eval), (int, float))
            ]
            sorted_valid_iterations = sorted(
                valid_iterations_for_sorting,
                key=lambda x: x["metrics"][selected_metric_eval],
                reverse=is_higher_better_metric
            )
            other_iterations = [
                res for res in iteration_results_for_task_display
                if res not in valid_iterations_for_sorting
            ]
            final_sorted_iterations_to_display = sorted_valid_iterations + other_iterations
            for iter_data in final_sorted_iterations_to_display:
                iter_num_display = iter_data['iteration_num']
                expander_title_str = f"Iteration {iter_num_display}"
                if "Error" in iter_data["metrics"]:
                    expander_title_str += " (Failed)"
                elif selected_metric_eval in iter_data["metrics"] and isinstance(iter_data["metrics"][selected_metric_eval], (int,float)):
                    metric_val_display = iter_data['metrics'][selected_metric_eval]
                    expander_title_str += f" - {selected_metric_eval}: {metric_val_display:.2f}"
                else:
                    expander_title_str += f" - {selected_metric_eval}: N/A"
                with st.expander(expander_title_str):
                    st.markdown("##### 🔧 Hyperparameters")
                    st.json(iter_data["params"])
                    st.markdown("##### 📊 Metrics")
                    if "Error" in iter_data["metrics"]:
                        st.error(f"This iteration failed: {iter_data['metrics']['Error']}")
                        continue
                    display_metrics_data = {k: v for k, v in iter_data["metrics"].items() if k not in ["Confusion Matrix", "Parameters"]}
                    num_metrics_cols_disp = min(len(display_metrics_data), 3)
                    if num_metrics_cols_disp > 0:
                        metric_cols_disp = st.columns(num_metrics_cols_disp)
                        col_idx_disp = 0
                        for metric_name_disp, value_disp in display_metrics_data.items():
                            current_col_disp = metric_cols_disp[col_idx_disp % num_metrics_cols_disp]
                            if isinstance(value_disp, (int, float, np.float_)):
                                current_col_disp.metric(metric_name_disp, f"{value_disp:.4f}")
                            else:
                                current_col_disp.metric(metric_name_disp, str(value_disp))
                            col_idx_disp += 1
                    if "Confusion Matrix" in iter_data["metrics"] and target_type == "Classification":
                        st.markdown("###### Confusion Matrix")
                        cm_data_plot = np.array(iter_data["metrics"]["Confusion Matrix"])
                        fig_cm_plot, ax_cm_plot = plt.subplots(figsize=(4,3))
                        sns.heatmap(cm_data_plot, annot=True, fmt="d", cmap="Blues", ax=ax_cm_plot, cbar=False)
                        ax_cm_plot.set_xlabel("Predicted")
                        ax_cm_plot.set_ylabel("Actual")
                        st.pyplot(fig_cm_plot)
                        plt.close(fig_cm_plot)
                    if iter_data["model_object"] and X_test.shape[0] > 0 and X_train.shape[0] > 0:
                        st.markdown("###### SHAP Summary Plot (on Test Set)")
                        try:
                            model_to_explain_shap = iter_data["model_object"]
                            if isinstance(model_to_explain_shap, (RandomForestClassifier, XGBClassifier, LGBMClassifier)) or \
                               (hasattr(model_to_explain_shap, '_estimator_type') and model_to_explain_shap._estimator_type == "regressor" and not isinstance(model_to_explain_shap, LinearRegression)):
                                explainer_shap = shap.Explainer(model_to_explain_shap, X_train)
                            else:
                                masker_shap = shap.maskers.Independent(X_train, max_samples=100)
                                explainer_shap = shap.Explainer(model_to_explain_shap, masker_shap)
                            shap_values_plot = explainer_shap(X_test)
                            fig_shap_plot, ax_shap_plot = plt.subplots()
                            if isinstance(shap_values_plot, list) and len(shap_values_plot) == 2:
                                shap.summary_plot(shap_values_plot[1], X_test, plot_type="dot", show=False, plot_size=None)
                            elif hasattr(shap_values_plot, "values"):
                                shap.summary_plot(shap_values_plot, X_test, plot_type="dot", show=False, plot_size=None)
                            else:
                                shap.summary_plot(shap_values_plot, X_test, plot_type="dot", show=False, plot_size=None)
                            st.pyplot(fig_shap_plot)
                            plt.close(fig_shap_plot)
                        except Exception as e_shap:
                            st.warning(f"⚠️ Could not generate SHAP summary plot: {e_shap}")

    # --- Select Iteration and Save Data/Metadata ---
    if feature_columns and iteration_results_for_task_display:
        st.subheader(f"💾 Select Best Iteration for {selected_model_task_name}")
        selectable_iterations_data = []
        if is_mvp:
            # In MVP, only one result, auto-select it
            for idx_iter_loop, res_iter_loop in enumerate(iteration_results_for_task_display):
                if "Error" not in res_iter_loop["metrics"] and res_iter_loop["model_object"] is not None:
                    display_name_iter_loop = f"Iteration {res_iter_loop['iteration_num']}"
                    selectable_iterations_data.append((display_name_iter_loop, idx_iter_loop))
        else:
            for idx_iter_loop, res_iter_loop in enumerate(iteration_results_for_task_display):
                if "Error" not in res_iter_loop["metrics"] and res_iter_loop["model_object"] is not None:
                    metric_val_iter_loop = res_iter_loop['metrics'].get(selected_metric_eval, "N/A")
                    if isinstance(metric_val_iter_loop, (int, float)):
                        metric_val_iter_loop = f"{metric_val_iter_loop:.2f}"
                    display_name_iter_loop = f"Iteration {res_iter_loop['iteration_num']} ({selected_metric_eval}: {metric_val_iter_loop})"
                    selectable_iterations_data.append((display_name_iter_loop, idx_iter_loop))
        if selectable_iterations_data:
            iteration_options_display_names = [item[0] for item in selectable_iterations_data]
            default_selection_display_idx = 0
            if "selected_iteration_global_index" in model_config:
                prev_selected_original_idx = model_config["selected_iteration_global_index"]
                for i_selectable, (_, original_idx_selectable) in enumerate(selectable_iterations_data):
                    if original_idx_selectable == prev_selected_original_idx:
                        default_selection_display_idx = i_selectable
                        break
            # In MVP, auto-select the only available result
            if is_mvp:
                selected_original_idx_confirm = selectable_iterations_data[0][1]
                selected_iteration_details_confirm = iteration_results_for_task_display[selected_original_idx_confirm]
                if st.button(f"Confirm Selection for {selected_model_task_name} (Iteration {selected_iteration_details_confirm['iteration_num']})", key=f"confirm_{task_key}"):
                    test_indices_for_iteration_save = selected_iteration_details_confirm.get("test_indices")
                    if test_indices_for_iteration_save is None or not isinstance(test_indices_for_iteration_save, list) or len(test_indices_for_iteration_save) == 0:
                        st.error("Could not retrieve valid test set indices for this iteration. Cannot save test data.")
                    else:
                        try:
                            columns_to_save_final = list(feature_columns) + [target_column] + timestamp_cols
                            columns_to_save_final = [col for col in columns_to_save_final if col in df_full.columns]
                            df_test_slice_save = df_full.loc[test_indices_for_iteration_save, columns_to_save_final].copy()
                            output_file_name_save = f"{target_column}_test_data_task_{task_key.replace(' ', '_').replace('(', '').replace(')', '')}_iter_{selected_iteration_details_confirm['iteration_num']}.parquet"
                            output_data_dir = os.path.join("data_registry", "selected_iterations")
                            os.makedirs(output_data_dir, exist_ok=True)
                            output_file_path_save = os.path.join(output_data_dir, output_file_name_save)
                            df_test_slice_save.to_parquet(output_file_path_save, index=False)
                            st.session_state.confirmed_model_outputs[task_key] = {
                                "task_name": selected_model_task_name,
                                "target_variable": target_column,
                                "selected_features": feature_columns,
                                "model_name": selected_model,
                                "hyperparameters": selected_iteration_details_confirm["params"],
                                "test_data_path": output_file_path_save,
                                "key_metrics": {k: v for k, v in selected_iteration_details_confirm["metrics"].items() if not (isinstance(v, (np.ndarray, list)) or k == "Error" or k == "Parameters")},
                                "iteration_number": selected_iteration_details_confirm['iteration_num'],
                            }
                            model_config["selected_iteration_global_index"] = selected_original_idx_confirm
                            st.session_state.model_development_state[task_key] = model_config
                        except Exception as save_e_final:
                            st.error(f"Error creating/saving test data or updating session state for {selected_model_task_name}: {save_e_final}")
                            st.exception(save_e_final)
            else:
                selected_iteration_display_name_confirm = st.selectbox(
                    f"Select Best Iteration for {selected_model_task_name}",
                    iteration_options_display_names,
                    index=default_selection_display_idx,
                    key=f"iteration_select_{task_key}"
                )
                selected_original_idx_confirm = next(orig_idx for name, orig_idx in selectable_iterations_data if name == selected_iteration_display_name_confirm)
                selected_iteration_details_confirm = iteration_results_for_task_display[selected_original_idx_confirm]
                if st.button(f"Confirm Selection for {selected_model_task_name} (Iteration {selected_iteration_details_confirm['iteration_num']})", key=f"confirm_{task_key}"):
                    test_indices_for_iteration_save = selected_iteration_details_confirm.get("test_indices")
                    if test_indices_for_iteration_save is None or not isinstance(test_indices_for_iteration_save, list) or len(test_indices_for_iteration_save) == 0:
                        st.error("Could not retrieve valid test set indices for this iteration. Cannot save test data.")
                    else:
                        try:
                            columns_to_save_final = list(feature_columns) + [target_column] + timestamp_cols
                            columns_to_save_final = [col for col in columns_to_save_final if col in df_full.columns]
                            df_test_slice_save = df_full.loc[test_indices_for_iteration_save, columns_to_save_final].copy()
                            output_file_name_save = f"{target_column}_test_data_task_{task_key.replace(' ', '_').replace('(', '').replace(')', '')}_iter_{selected_iteration_details_confirm['iteration_num']}.parquet"
                            output_data_dir = os.path.join("data_registry", "selected_iterations")
                            os.makedirs(output_data_dir, exist_ok=True)
                            output_file_path_save = os.path.join(output_data_dir, output_file_name_save)
                            df_test_slice_save.to_parquet(output_file_path_save, index=False)
                            st.session_state.confirmed_model_outputs[task_key] = {
                                "task_name": selected_model_task_name,
                                "target_variable": target_column,
                                "selected_features": feature_columns,
                                "model_name": selected_model,
                                "hyperparameters": selected_iteration_details_confirm["params"],
                                "test_data_path": output_file_path_save,
                                "key_metrics": {k: v for k, v in selected_iteration_details_confirm["metrics"].items() if not (isinstance(v, (np.ndarray, list)) or k == "Error" or k == "Parameters")},
                                "iteration_number": selected_iteration_details_confirm['iteration_num'],
                            }
                            model_config["selected_iteration_global_index"] = selected_original_idx_confirm
                            st.session_state.model_development_state[task_key] = model_config
                        except Exception as save_e_final:
                            st.error(f"Error creating/saving test data or updating session state for {selected_model_task_name}: {save_e_final}")
                            st.exception(save_e_final)
        else:
            st.info("No successful model iterations are available to select for confirmation for this task.")

    st.session_state.model_development_state[task_key] = model_config

    # --- Proceed to Next Model Button (appears after confirmation) ---
    if task_key in st.session_state.get("confirmed_model_outputs", {}):
        st.success(f"✅ Configuration for {model_name} has been confirmed.")
        if model_index < len(modeling_tasks) - 1:
            next_model_name = modeling_tasks[model_index + 1]
            if st.button(f"Proceed to {next_model_name}", key=f"proceed_{task_key}"):
                # Show the next model section
                st.session_state["model_dev_visible_sections"][next_model_name] = True
                st.query_params["scroll_to"] = f"{next_model_name.replace(' ', '_')}"
                st.info(f"Scroll down to configure: {next_model_name}")
        
    else:
        st.warning(f"Please confirm the selection for {model_name} to proceed.")

# --- Finalize and Overview ---
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.subheader("Confirmed Model Selections")
confirmed_count_final = len(st.session_state.get("confirmed_model_outputs", {}))
all_tasks_count_final = len(modeling_tasks)
if confirmed_count_final > 0:
    st.write("Confirmed Model Selections Overview:")
    for task_name_key_final, conf_output_final in st.session_state.get("confirmed_model_outputs", {}).items():
        iter_num_final = conf_output_final.get("iteration_number", "N/A")
        model_name_final = conf_output_final.get("model_name", "N/A")
        target_var_final = conf_output_final.get("target_variable", "N/A")
        test_data_path_final = conf_output_final.get("test_data_path", "N/A")
        st.markdown(f"- **{conf_output_final['task_name']}**: Iteration `{iter_num_final}` for target `{target_var_final}` (Model: `{model_name_final}`). ")

unconfirmed_tasks = [
    task for task in modeling_tasks
    if f"{task.replace(' ', '_')}_task" not in st.session_state.get("confirmed_model_outputs", {})
]

# Place the "Model Development Completed" button here, after the confirmed selections section
if confirmed_count_final == all_tasks_count_final:
    st.success("✅ You can now proceed to Performance Monitoring page.")
    if st.button("Model Development Completed ! Proceed to Performance Monitoring", key="finalize_model_dev"):
        st.session_state["model_development_complete"] = True
        st.info("Model development step completed. You can now proceed to Performance Monitoring.")
        st.switch_page("pages/Performance Monitoring.py")