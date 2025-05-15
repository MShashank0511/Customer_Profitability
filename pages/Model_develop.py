import numpy as np
import os
import json
import shutil # Added for potential directory cleanup
import streamlit as st
# --- Streamlit Page Configuration ---


# Patch for deprecated np.bool - Use np.bool_ directly if possible
# In modern NumPy versions, np.bool is deprecated. Use np.bool_
# Removing the patch as it seems unnecessary based on the code's usage
# If you encounter issues with older libraries requiring np.bool,
# you might need a more specific compatibility handling depending on the library.
# For now, let's assume np.bool_ is available and sufficient.
# if not hasattr(np, 'bool'):
#     np.bool = np.bool_

from xgboost import XGBClassifier
import plotly.express as px
import streamlit as st
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
import joblib # To potentially save the model itself if needed later

st.title("🔧 Model Development")

# --- Data Loading from Session State ---
data_path = st.session_state.get("on_us_data_path")
if not data_path or not os.path.exists(data_path):
    st.error("No registered data file found. Please upload and register data in the Data Engineering step first.")
    st.stop()

try:
    df_full = pd.read_parquet(data_path)
except Exception as e:
    st.error(f"Error loading data from {data_path}: {e}")
    st.stop()

# Drop 'timestamp' column if it exists, keep 'Timestamp'
if 'timestamp' in df_full.columns:
    df_full = df_full.drop(columns=['timestamp'])

# Define target variables and check their existence
target_variables_list = ["Profitability_GBP", "COF_EVENT_LABEL", "PREPAYMENT_EVENT_LABEL"]
for target in target_variables_list:
    if target not in df_full.columns:
        st.error(f"Target variable '{target}' not found in the dataset. Please check your data in the Data Engineering step.")
        st.stop()

# List of modeling tasks
modeling_tasks = [
    {"name": "Model 1 (Profitability_GBP)", "target": "Profitability_GBP"},
    {"name": "Model 2 (COF_EVENT_LABEL)", "target": "COF_EVENT_LABEL"},
    {"name": "Model 3 (PREPAYMENT_EVENT_LABEL)", "target": "PREPAYMENT_EVENT_LABEL"},
]

# --- Session State Initialization ---
# This will store the selections made for each model task within this page
if "model_development_state" not in st.session_state:
    st.session_state.model_development_state = {}

# This will store the *final confirmed* output (data path and model metadata) for each task
if "confirmed_model_outputs" not in st.session_state:
    st.session_state.confirmed_model_outputs = {}

# --- Setup Output Directory ---
# Directory to save the selected test data subsets
output_data_dir = "data_registry/selected_iteration_data"
os.makedirs(output_data_dir, exist_ok=True)

# --- Step 1: Dropdown for Model Selection ---
model_task_options = [task["name"] for task in modeling_tasks]
selected_model_task_name = st.selectbox(
    "Select Model Task to Work On",
    model_task_options
)
current_task_config = next(task for task in modeling_tasks if task["name"] == selected_model_task_name)
target_column = current_task_config["target"]
task_key = selected_model_task_name # Use the full name as the key for session state

# Load state for the current task
current_task_state = st.session_state.model_development_state.get(task_key, {})

# Work on a copy of the full dataframe
df = df_full.copy()

# --- Step 2: Sub-sampling ---
st.subheader("🔍 Sub-sampling")
sample_frac = st.slider(
    "Select sub-sample fraction", 0.01, 1.0, # Allow smaller fractions
    current_task_state.get("sample_frac", 1.0), key=f"sample_frac_{task_key}"
)
if sample_frac < 1.0:
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    st.info(f"Using a {sample_frac:.2f} fraction of the data ({len(df)} rows).")
else:
     st.info("Using the full dataset.")


# --- Step 3: Target Variable Selection & Distribution ---
st.subheader("🎯 Target Variable Analysis")
# Check for typical classification data types and number of unique values
if df[target_column].nunique() < 10 and df[target_column].dtype in ['int64', 'float64', 'object', 'category', 'bool']:
    target_type = "Classification"
    # Ensure target is numeric for classification metrics and model training
    if df[target_column].dtype == 'object' or df[target_column].dtype == 'category' or df[target_column].dtype == 'bool':
         try:
             # Attempt to convert to numeric (binary 0/1 or multi-class)
             original_values = df[target_column].unique()
             if len(original_values) <= 2: # Binary classification - encode to 0 and 1
                  le = LabelEncoder()
                  df[target_column] = le.fit_transform(df[target_column])
                  st.info(f"Encoded categorical target '{target_column}' using LabelEncoder.")
                  st.write(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
             else: # Multi-class - encode if not numeric
                  st.warning(f"Multi-class classification target '{target_column}' detected. Some metrics and SHAP plots may behave differently.")
                  if not pd.api.types.is_numeric_dtype(df[target_column]):
                       le = LabelEncoder()
                       df[target_column] = le.fit_transform(df[target_column])

         except Exception as e:
              st.error(f"Could not encode target variable '{target_column}': {e}")
              st.stop() # Stop if target cannot be processed for classification

else:
    target_type = "Regression"

st.markdown(f"**Target Variable:** {target_column}")
st.markdown(f"**Inferred Problem Type:** {target_type}")

if target_type == "Regression":
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[target_column], kde=True, ax=ax, color="blue")
    ax.set_title(f"Distribution of {target_column}")
    ax.set_xlabel(target_column)
    st.pyplot(fig)
    plt.close(fig) # Close figure
else: # Classification
    try:
        # Ensure target is 0 and 1 for binary classification metrics where applicable
        unique_targets = df[target_column].unique()
        if len(unique_targets) == 2:
             # Calculate event rate for the positive class (assuming 1 after encoding)
             event_rate_value = df[target_column].value_counts(normalize=True).get(1, 0) * 100
             st.metric(
                 label="Event Rate (%)",
                 value=f"{event_rate_value:.2f}%",
                 help="Percentage of positive events (class 1) in the target variable."
             )
        else:
             st.info(f"Target '{target_column}' has {len(unique_targets)} unique values. Displaying frequency.")


        frequency_df = df[target_column].value_counts().reset_index()
        frequency_df.columns = [target_column, "count"]
        fig = px.bar(
            frequency_df,
            x=target_column,
            y="count",
            text="count",
            labels={target_column: target_column, "count": "Frequency"},
            title=f"Frequency of {target_column}"
        )
        fig.update_traces(textposition="inside")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying target distribution: {e}")


# --- Step 4: Feature Selection & Train/Test Split ---
st.subheader("📊 Select Features")

# Exclude target, Timestamp, and TERM_OF_LOAN from selectable features but keep them for data export
excluded_cols_from_features = target_variables_list + ["Timestamp", "TERM_OF_LOAN"]
selectable_features = [col for col in df.columns if col not in excluded_cols_from_features]

# Restore previously selected features if available, default to all selectable
selected_features = current_task_state.get("feature_columns", selectable_features)

feature_columns = st.multiselect(
    "Select features to include in the model",
    options=selectable_features,
    default=selected_features,
    key=f"features_{task_key}"
)

# Ensure the current state is saved
current_task_state["feature_columns"] = feature_columns
st.session_state.model_development_state[task_key] = current_task_state


if not feature_columns:
    st.warning("Please select at least one feature to proceed.")
else:
    test_size = st.slider(
        "Select test size", 0.01, 0.5, # Allow smaller test sizes
        current_task_state.get("test_size", 0.2), key=f"test_size_{task_key}"
    )
    current_task_state["test_size"] = test_size
    st.session_state.model_development_state[task_key] = current_task_state

    X = df[feature_columns]
    y = df[target_column]

    # Check if target is suitable for stratification
    stratify_y = None
    if target_type == "Classification":
         unique_classes, counts = np.unique(y, return_counts=True)
         # Stratify only if there's more than one class and each class has at least 2 samples (required by stratify)
         if len(unique_classes) > 1 and np.min(counts) >= 2:
              stratify_y = y
         elif len(unique_classes) > 1:
             st.warning(f"Stratification skipped: Some classes in target '{target_column}' have fewer than 2 samples.")


    # Store original indices to retrieve test set rows later
    indices = df.index
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=test_size, random_state=42, stratify=stratify_y
    )

    st.info(f"Train set size: {len(X_train)} rows")
    st.info(f"Test set size: {len(X_test)} rows")


    # --- Step 5: Model Selection ---
    st.subheader("📚 Select Model")
    if target_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression,
            "LGBM Classifier": LGBMClassifier,
            "Random Forest Classifier": RandomForestClassifier,
            "XGBoost Classifier": XGBClassifier
        }
    else:
        models = {"Linear Regression": LinearRegression}

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        index=list(models.keys()).index(current_task_state.get("selected_model", list(models.keys())[0])),
        key=f"model_{task_key}"
    )
    current_task_state["selected_model"] = selected_model
    st.session_state.model_development_state[task_key] = current_task_state
    model_class = models[selected_model]

    # --- Step 6: Run Model & Hyperparameter Tuning ---
    if st.button(f"Run Model for {selected_model_task_name}", key=f"run_{task_key}"):
        st.subheader("🏆 Hyperparameter Iterations")
        param_dist = {
            "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear"]}, # Added 0.01
            "LGBM Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7, 9], "learning_rate": [0.01, 0.05, 0.1, 0.2]}, # Added 200, 9, 0.05
            "Random Forest Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7, 9]}, # Added 200, 9
            "XGBoost Classifier": {"n_estimators": [50, 100, 150, 200], "max_depth": [3, 5, 7, 9], "learning_rate": [0.01, 0.05, 0.1, 0.2], 'use_label_encoder': [False], 'eval_metric': ['logloss'] if target_type == 'Classification' else ['rmse']}, # Added 200, 9, 0.05
            "Linear Regression": {"fit_intercept": [True, False]}
        }

        if selected_model not in param_dist:
             st.error(f"Parameter distribution not defined for {selected_model}")
             # Clear previous results if model selection changes and run button is hit
             if f"iteration_results_{task_key}" in st.session_state:
                  del st.session_state[f"iteration_results_{task_key}"]
             st.stop()

        # Number of iterations to sample
        n_iterations = 5 # Can be adjusted
        try:
            sampled_params = list(ParameterSampler(param_dist[selected_model], n_iter=n_iterations, random_state=42))
        except ValueError as e:
             st.error(f"Error sampling parameters: {e}. Check your parameter distribution ranges and n_iter.")
             # Clear previous results
             if f"iteration_results_{task_key}" in st.session_state:
                  del st.session_state[f"iteration_results_{task_key}"]
             st.stop()

        iteration_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, params in enumerate(sampled_params):
            status_text.text(f"Running iteration {i+1}/{n_iterations}...")
            progress_bar.progress((i + 1) / n_iterations)

            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = {}

                if target_type == "Classification":
                    # Ensure y_test is binary (0 or 1) for binary metrics
                    unique_test_classes = np.unique(y_test)
                    if len(unique_test_classes) == 2:
                         # Binary Classification Metrics
                         y_proba = None
                         if hasattr(model, "predict_proba"):
                              try:
                                   y_proba = model.predict_proba(X_test)
                                   # predict_proba output shape is (n_samples, n_classes).
                                   # We need the probability of the positive class (usually index 1 after Label Encoding)
                                   # Check if y_test contains both classes before computing ROC/Gini
                                   if len(unique_test_classes) == 2:
                                        y_proba_positive = y_proba[:, 1]
                                        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba_positive)
                                        metrics["Gini Index"] = 2 * metrics["ROC-AUC"] - 1
                                   else:
                                        st.info("ROC-AUC/Gini not computable: Test set contains only one class.")
                                        metrics["ROC-AUC"] = "N/A (Single Class)"
                                        metrics["Gini Index"] = "N/A (Single Class)"

                              except Exception as proba_e:
                                   st.warning(f"Could not compute probabilities for ROC/Gini: {proba_e}")
                                   metrics["ROC-AUC"] = "N/A (Error)"
                                   metrics["Gini Index"] = "N/A (Error)"
                         else:
                              st.info("Model does not have predict_proba method for ROC/Gini.")
                              metrics["ROC-AUC"] = "N/A (No Proba)"
                              metrics["Gini Index"] = "N/A (No Proba)"

                         # Compute binary classification metrics (y_pred must be 0 or 1)
                         try:
                              metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                              metrics["Precision"] = precision_score(y_test, y_pred, average='binary', zero_division=0)
                              metrics["Recall"] = recall_score(y_test, y_pred, average='binary', zero_division=0)
                              metrics["F1 Score"] = f1_score(y_test, y_pred, average='binary', zero_division=0)
                              metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
                         except Exception as bin_metrics_e:
                              st.warning(f"Could not compute binary metrics: {bin_metrics_e}")
                              metrics["Accuracy"] = metrics["Precision"] = metrics["Recall"] = metrics["F1 Score"] = "Error"
                              metrics["Confusion Matrix"] = [[0,0],[0,0]] # Placeholder


                    else: # Multi-class or other classification type
                         # Multi-class Metrics
                         metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                         # Add other multi-class metrics if needed (e.g., macro/weighted average)
                         st.info("Multi-class target detected. Displaying Accuracy (Weighted Precision, Recall, F1).")
                         try:
                              metrics["Precision (Weighted)"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                              metrics["Recall (Weighted)"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                              metrics["F1 Score (Weighted)"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                         except Exception as multi_metrics_e:
                              st.warning(f"Could not compute weighted multi-class metrics: {multi_metrics_e}")
                              metrics["Precision (Weighted)"] = metrics["Recall (Weighted)"] = metrics["F1 Score (Weighted)"] = "Error"


                         metrics["ROC-AUC"] = "N/A (Multi-class)"
                         metrics["Gini Index"] = "N/A (Multi-class)"
                         try:
                            metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
                         except Exception as cm_e:
                             st.warning(f"Could not compute Confusion Matrix: {cm_e}")
                             metrics["Confusion Matrix"] = "Error"


                else: # Regression
                    metrics = {
                        "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_test, y_pred)),
                    }
                    try:
                         # Calculate MAPE, handle division by zero if actuals contain zero
                         # Add a small epsilon to avoid division by zero
                         mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100 # Added epsilon
                         if np.isfinite(mape):
                              metrics["Mean Absolute Percentage Error (MAPE)"] = mape
                         else:
                              metrics["Mean Absolute Percentage Error (MAPE)"] = "Inf/NaN (Actuals include zero)"
                    except Exception as mape_e:
                         metrics["Mean Absolute Percentage Error (MAPE)"] = f"Error: {mape_e}"


                iteration_results.append({"params": params, "metrics": metrics, "model": model, "test_indices": test_indices})

            except Exception as e:
                st.warning(f"⚠️ Skipping iteration {i+1} due to error: {e}")
                iteration_results.append({"params": params, "metrics": {"Error": str(e)}, "model": None, "test_indices": None})


        st.session_state[f"iteration_results_{task_key}"] = iteration_results
        status_text.text("Model iterations complete.")
        progress_bar.empty()

        # Display Results
        st.subheader("📊 Iteration Results")
        if not iteration_results:
             st.info("No model iterations were run.")
        else:
             for i, iteration in enumerate(iteration_results):
                 with st.expander(f"Iteration {i+1}"):
                     if "Error" in iteration["metrics"]:
                         st.error(f"This iteration failed: {iteration['metrics']['Error']}")
                         continue

                     st.markdown("### 🔧 Hyperparameters")
                     st.json(iteration["params"])

                     st.markdown("### 📊 Metrics")

                     if target_type == "Classification":
                         # Display classification metrics
                         metrics_cols = st.columns(6)
                         metrics_to_display = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Gini Index"]
                         for j, metric_name in enumerate(metrics_to_display):
                             value = iteration['metrics'].get(metric_name, 'N/A')
                             # Format only if it's a number, otherwise display as is
                             display_value = f"{value:.4f}" if isinstance(value, (int, float, np.float_)) else value
                             metrics_cols[j].metric(metric_name, display_value)

                         # Confusion Matrix
                         if isinstance(iteration["metrics"].get("Confusion Matrix"), np.ndarray):
                             st.markdown("#### Confusion Matrix")
                             try:
                                 fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                                 sns.heatmap(iteration["metrics"]["Confusion Matrix"], ax=ax_cm, annot=True, fmt="d", cmap="Blues")
                                 ax_cm.set_xlabel("Predicted")
                                 ax_cm.set_ylabel("Actual")
                                 st.pyplot(fig_cm)
                                 plt.close(fig_cm)
                             except Exception as cm_plot_e:
                                  st.warning(f"Could not plot Confusion Matrix: {cm_plot_e}")


                         # SHAP Summary Plot
                         st.markdown("#### SHAP Summary Plot")
                         # SHAP for tree models and linear models is supported
                         if iteration["model"] is not None and selected_model in ["LGBM Classifier", "Random Forest Classifier", "XGBoost Classifier", "Logistic Regression"]:
                             try:
                                # Use a subset of the test data for SHAP to speed up calculation
                                # Ensure the subset size is not larger than the test set size
                                shap_sample_size = min(500, X_test.shape[0])
                                if shap_sample_size > 0:
                                    # Use random.choice for indices to handle potential non-contiguous index
                                    shap_indices_sample = np.random.choice(X_test.index, shap_sample_size, replace=False)
                                    X_test_shap = X_test.loc[shap_indices_sample]


                                    plt.clf()  # Clear the current figure context
                                    explainer = shap.Explainer(iteration["model"], X_test_shap) # Explain on sample
                                    shap_values = explainer(X_test_shap) # Calculate on sample

                                    # Handle SHAP values structure for binary vs multi-class
                                    if target_type == "Classification" and len(np.unique(y_test)) == 2 and isinstance(shap_values, list) and len(shap_values) > 1:
                                         # For binary classification, summary_plot often expects values for one class
                                         # Assuming class 1 is the positive class after encoding
                                         shap_values = shap_values[1]
                                    elif target_type == "Classification" and len(np.unique(y_test)) > 2 and isinstance(shap_values, list):
                                         # For multi-class, you might plot values for a specific class,
                                         # or loop through classes. Default summary plot can handle list.
                                         pass # Use the list of SHAP values


                                    shap.summary_plot(shap_values, X_test_shap, show=False)
                                    fig = plt.gcf() # Get the current figure after SHAP plot is generated
                                    st.pyplot(fig)
                                    plt.close(fig) # Close the figure to free memory
                                else:
                                     st.info("Test set is too small to generate SHAP plot.")
                             except Exception as shap_e:
                                 st.warning(f"⚠️ Could not generate SHAP summary plot for this iteration: {shap_e}")
                         else:
                              st.info(f"SHAP summary plot not standard or available for {selected_model}.")


                     elif target_type == "Regression":
                         # Display regression metrics
                         metrics_cols = st.columns(3)
                         metrics_to_display = ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)"]
                         for j, metric_name in enumerate(metrics_to_display):
                              value = iteration['metrics'].get(metric_name, 'N/A')
                              display_value = f"{value:.4f}" if isinstance(value, (int, float, np.float_)) else value
                              metrics_cols[j].metric(metric_name, display_value)


                         # SHAP Summary Plot
                         st.markdown("#### SHAP Summary Plot")
                         if iteration["model"] is not None and selected_model in ["LGBM Classifier", "Random Forest Classifier", "XGBoost Classifier", "Linear Regression"]:
                             try:
                                plt.clf()  # Clear current figure context
                                # Use a subset of the test data for SHAP
                                shap_sample_size = min(500, X_test.shape[0])
                                if shap_sample_size > 0:
                                    shap_indices_sample = np.random.choice(X_test.index, shap_sample_size, replace=False)
                                    X_test_shap = X_test.loc[shap_indices_sample]

                                    explainer = shap.Explainer(iteration["model"], X_test_shap)
                                    shap_values = explainer(X_test_shap)

                                    shap.summary_plot(shap_values, X_test_shap, show=False, color_bar=True) # Added color_bar for regression
                                    fig = plt.gcf()  # Get the current figure after SHAP plot is generated
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                     st.info("Test set is too small to generate SHAP plot.")
                             except Exception as shap_e:
                                 st.warning(f"⚠️ Could not generate SHAP summary plot for this iteration: {shap_e}")
                         else:
                             st.info(f"SHAP summary plot not standard or available for {selected_model}.")


    # --- Step 7: Select Iteration and Save Data/Metadata ---
    iteration_results = st.session_state.get(f"iteration_results_{task_key}", [])
    if iteration_results:
        # Filter out failed iterations for selection
        successful_iterations = [i for i, res in enumerate(iteration_results) if "Error" not in res["metrics"]]

        if successful_iterations:
            iteration_options = [f"Iteration {i+1}" for i in successful_iterations]
            # Get previously selected index relative to successful_iterations
            prev_selected_idx_global = current_task_state.get("selected_iteration_index", successful_iterations[0] if successful_iterations else 0) # Default to first successful
            try:
                 # Find the index in the *filtered* list that corresponds to the previously selected global index
                 prev_selected_idx_filtered = successful_iterations.index(prev_selected_idx_global)
            except ValueError:
                 # If previous index is no longer valid, default to first successful
                 prev_selected_idx_filtered = 0


            selected_iteration_option = st.selectbox(
                f"Select Best Iteration for {selected_model_task_name}",
                iteration_options,
                index=prev_selected_idx_filtered,
                key=f"iteration_select_{task_key}"
            )

            # Get the original index of the selected iteration from the full list
            # The option is "Iteration X", so split and convert X-1
            selected_iteration_global_index = int(selected_iteration_option.split(" ")[1]) - 1
            iteration_details = iteration_results[selected_iteration_global_index]

            if st.button(f"Confirm Selection for {selected_model_task_name}", key=f"confirm_{task_key}"):
                 if "Error" in iteration_details["metrics"]:
                      st.error("Cannot confirm selection for a failed iteration.")
                 else:
                      # Get the test set indices for this iteration's split
                      test_indices_for_iteration = iteration_details["test_indices"]

                      if test_indices_for_iteration is None or len(test_indices_for_iteration) == 0:
                           st.error("Could not retrieve test set data for this iteration.")
                      else:
                           # Create the DataFrame slice for the test set based on selected features + essentials
                           columns_to_save = list(feature_columns) + [target_column] # Make a copy of feature_columns
                           if "Timestamp" in df_full.columns and "Timestamp" not in columns_to_save:
                               columns_to_save.append("Timestamp")
                           if "TERM_OF_LOAN" in df_full.columns and "TERM_OF_LOAN" not in columns_to_save:
                               columns_to_save.append("TERM_OF_LOAN")

                           # Select rows from the *original sampled* DataFrame (`df`) using test indices
                           # This ensures we get Timestamp and TERM_OF_LOAN even if not features
                           try:
                               df_test_slice = df.loc[test_indices_for_iteration, columns_to_save].copy()

                               # Define the output file path
                               # Using a consistent naming convention
                               output_file_name = f"{target_column}_test_data.parquet"
                               output_file_path = os.path.join(output_data_dir, output_file_name)

                               # Save the test data slice to a Parquet file
                               df_test_slice.to_parquet(output_file_path, index=False)
                               st.success(f"✅ Test data for {selected_model_task_name} saved to: {output_file_path}")

                               # Store the confirmed configuration in session state for the next page
                               st.session_state.confirmed_model_outputs[task_key] = {
                                   "task_name": selected_model_task_name,
                                   "target_variable": target_column,
                                   "selected_features": feature_columns, # Store features list explicitly
                                   "model_name": selected_model,
                                   "hyperparameters": iteration_details["params"],
                                   "test_data_path": output_file_path,
                                   # Optionally store key metrics for quick display on the next page
                                   "key_metrics": {k: v for k, v in iteration_details["metrics"].items() if not (isinstance(v, np.ndarray) or k == "Error")} # Exclude arrays like Confusion Matrix
                               }

                               # Update the development state to remember which iteration was selected
                               current_task_state["selected_iteration_index"] = selected_iteration_global_index
                               st.session_state.model_development_state[task_key] = current_task_state

                               st.success(f"✅ Configuration for {selected_model_task_name} (Iteration {selected_iteration_global_index + 1}) confirmed and saved!")

                           except Exception as save_e:
                               st.error(f"Error creating/saving test data or updating session state: {save_e}")

        else:
             st.info("No successful iterations available to select.")


# --- Step 8: Finalize and Proceed ---
st.markdown("---")
st.subheader("Finalize and Proceed")

# Check if all modeling tasks have a confirmed output
# Count how many tasks have an entry in confirmed_model_outputs
confirmed_count = len(st.session_state.confirmed_model_outputs)
all_tasks_count = len(modeling_tasks)

if confirmed_count == all_tasks_count:
    st.success(f"✅ All {all_tasks_count} model tasks have a confirmed selection. You can now proceed to the next page.")
    # You would typically have a button here to navigate, managed by your multi-page app structure
    # For demonstration, a placeholder button:
    if st.button("Proceed to Next Page"):
         # In a multi-page app, this button would trigger the page change
         st.info("Proceeding to the next page...")
         # Example of setting a state variable to indicate readiness
         st.session_state["model_development_complete"] = True
         # Logic to navigate to the next page would go here, e.g., st.switch_page("pages/back.py")
else:
    st.info(f"Please confirm a selected iteration for all {all_tasks_count} model tasks ({confirmed_count}/{all_tasks_count} confirmed) before proceeding.")
    # Optional: Display confirmed tasks for user
    # if confirmed_count > 0:
    #      st.write("Confirmed Tasks:")
    #      for task_name in st.session_state.confirmed_model_outputs.keys():
    #           st.write(f"- {task_name}")