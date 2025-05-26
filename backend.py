# back.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
# Import necessary model classes
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
# Import necessary metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score
)
from sklearn.preprocessing import LabelEncoder # For handling target encoding with default data
# Import train_test_split for splitting data
from sklearn.model_selection import train_test_split
import inspect # Needed in create_model_instance

# Path to the default data directory
DEFAULT_DATA_DIR = "default_data"

# Function to load default data if session state is empty
def load_default_data(): # Using your provided function name
    # st.write("back.py: Executing load_default_data function...") # Debugging line
    default_files = {
        "COF_EVENT_LABEL": os.path.join(DEFAULT_DATA_DIR, "default_COF_EVENT_LABEL_test_data.parquet"),
        "PREPAYMENT_EVENT_LABEL": os.path.join(DEFAULT_DATA_DIR, "default_PREPAYMENT_EVENT_LABEL_test_data.parquet"),
        "Profitability_GBP": os.path.join(DEFAULT_DATA_DIR, "default_Profitability_GBP_test_data.parquet"),
        "on_us_data": os.path.join(DEFAULT_DATA_DIR, "default_on_us_data.parquet"),
    }
    any_default_loaded = False
    for key, file_path in default_files.items():
        if key not in st.session_state or not isinstance(st.session_state.get(key), pd.DataFrame):
            if os.path.exists(file_path):
                try:
                    st.session_state[key] = pd.read_parquet(file_path)
                    # st.write(f"✔️ Default data for '{key}' loaded into session state by back.py from {file_path}.") # Debug
                    any_default_loaded = True
                except Exception as e:
                    st.warning(f"⚠️ Default file for {key} at {file_path} could not be loaded by back.py: {e}")
            elif key == "on_us_data":
                 st.warning(f"⚠️ Default main data file ('on_us_data.parquet') for '{key}' not found at {file_path}.")
    # if any_default_loaded:
        # st.info("back.py: Default data loading into session state complete/attempted.")


# Check if 'on_us_data' DataFrame is in session state. If not, try to load defaults.
# This is crucial for generate_default_confirmed_outputs.
if "on_us_data" not in st.session_state or not isinstance(st.session_state.get("on_us_data"), pd.DataFrame):
    # st.info("back.py: 'on_us_data' DataFrame not in session. Attempting to load default data files...") # Debug
    load_default_data()


# --- Helper to generate default "confirmed_model_outputs" ---
def generate_default_confirmed_outputs():
    default_outputs = {}
    modeling_tasks_configs = [
        {"name": "Model 1 (Profitability_GBP)", "target": "Profitability_GBP", "model_type": "Regression"},
        {"name": "Model 2 (COF_EVENT_LABEL)", "target": "COF_EVENT_LABEL", "model_type": "Classification"},
        {"name": "Model 3 (PREPAYMENT_EVENT_LABEL)", "target": "PREPAYMENT_EVENT_LABEL", "model_type": "Classification"},
    ]

    default_main_data_path = os.path.join(DEFAULT_DATA_DIR, "default_on_us_data.parquet")

    if "on_us_data" not in st.session_state or not isinstance(st.session_state["on_us_data"], pd.DataFrame):
        st.error("Cannot generate default model configurations: 'default_on_us_data.parquet' must be loaded and available in session state.")
        return {}
    
    df_on_us_default = st.session_state["on_us_data"] # This is the DataFrame

    all_potential_target_names = [task_cfg["target"] for task_cfg in modeling_tasks_configs]
    common_non_features = ["Timestamp_x", "timestamp", "ID", "AccountID", "account_id"] # Add any other known non-features

    # Derive a common list of features from default_on_us_data, excluding all targets and common non-features
    base_default_features = [
        col for col in df_on_us_default.columns 
        if col not in all_potential_target_names and col not in common_non_features
    ]

    if not base_default_features:
        st.warning("Could not derive any base default features from 'default_on_us_data.parquet' after excluding targets and common non-features. Default configurations might be unusable.")
        # You could define a hardcoded minimal list here as a last resort if needed:
        # base_default_features = ["some_very_common_feature1", "some_very_common_feature2"]

    for task_cfg in modeling_tasks_configs:
        task_key = task_cfg["name"]
        current_target = task_cfg["target"]

        if current_target not in df_on_us_default.columns:
            st.warning(f"Default target variable '{current_target}' for task '{task_key}' not found in 'default_on_us_data.parquet'. Skipping this default configuration.")
            continue
        
        # Features for this specific task should not include its own target (already handled by base_default_features logic)
        task_specific_default_features = base_default_features # Using the common derived list

        default_outputs[task_key] = {
            "task_name": task_key,
            "target_variable": current_target,
            "selected_features": task_specific_default_features,
            "test_data_path": default_main_data_path, # All default tasks use the main default Parquet file
            "model_name": "Linear Regression" if task_cfg["model_type"] == "Regression" else "Logistic Regression",
            "hyperparameters": {} if task_cfg["model_type"] == "Regression" else {"solver": "liblinear", "max_iter": 100}, # Minimal default params
            "key_metrics": {"Info": "Default configuration. Metrics from dev step are N/A."}, # Placeholder
            "iteration_number": 0, # Signifies a default configuration
        }
    
    if not default_outputs:
        st.warning("No default configurations could be generated. This might be due to missing targets or features in 'default_on_us_data.parquet'.")

    return default_outputs


# --- Helper to create model instances ---
def create_model_instance(model_name, hyperparameters):
    try:
        model_classes = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "XGBoost Classifier": XGBClassifier,
            "XGBoost Regressor": XGBRegressor,
            "LGBM Classifier": LGBMClassifier,
            "LGBM Regressor": LGBMRegressor,
            "Random Forest Classifier": RandomForestClassifier
        }
        model_class = model_classes.get(model_name)
        if model_class:
            valid_params = inspect.signature(model_class).parameters
            filtered_params = {k: v for k, v in hyperparameters.items() if k in valid_params}
            
            # Ensure XGBoost specific params are handled if model is XGBoost
            if model_name.startswith("XGB") and 'use_label_encoder' in valid_params:
                 filtered_params['use_label_encoder'] = False # Generally recommended to be False
                 # Add default eval_metric if not provided, based on classifier/regressor
                 if model_name == "XGBoost Classifier" and 'eval_metric' not in filtered_params:
                      filtered_params['eval_metric'] = 'logloss' 
                 elif model_name == "XGBoost Regressor" and 'eval_metric' not in filtered_params:
                      filtered_params['eval_metric'] = 'rmse'
            return model_class(**filtered_params)
        else:
            # st.warning(f"Model name '{model_name}' not recognized in create_model_instance.") # Avoid direct st calls in low-level helpers
            print(f"Warning: Model name '{model_name}' not recognized in create_model_instance.")
            return None
    except Exception as e:
        print(f"Error creating model instance {model_name}: {e}") # Print error instead of st.error
        return None

# --- Function to load and prepare data for the NEXT page (model_monitoring.py) ---
def process_models_from_session():
    confirmed_models_data = st.session_state.get("confirmed_model_outputs")
    source_of_config = "User Confirmed (Model Development)" # Default assumption

    if not confirmed_models_data:
        st.info("backend.py/process_models: No user-confirmed models found. Attempting to generate and use default configurations.")
        confirmed_models_data = generate_default_confirmed_outputs()
        source_of_config = "Default Configuration"
        if not confirmed_models_data:
            st.error("backend.py/process_models: Failed to generate default model configurations. Cannot proceed.")
            return []

    processed_dataframes_list = []
    for task_key_proc, task_info_proc in confirmed_models_data.items():
        try:
            test_data_path_proc = task_info_proc.get("test_data_path")
            target_variable_proc = task_info_proc.get("target_variable")
            selected_features_proc = task_info_proc.get("selected_features")
            model_name_proc = task_info_proc.get("model_name")
            hyperparams_proc = task_info_proc.get("hyperparameters")

            # Basic validation of retrieved info
            if not all([test_data_path_proc, target_variable_proc, isinstance(selected_features_proc, list), model_name_proc, isinstance(hyperparams_proc, dict)]):
                st.error(f"Task '{task_key_proc}' has incomplete or malformed configuration. Skipping.")
                continue
            if not os.path.exists(test_data_path_proc):
                st.error(f"Skipping task '{task_key_proc}': Source data file not found at {test_data_path_proc}")
                continue

            df_source_for_task = pd.read_parquet(test_data_path_proc)

            # Validate that features and target exist in the loaded DataFrame
            missing_feats_proc = [f for f in selected_features_proc if f not in df_source_for_task.columns]
            if missing_feats_proc:
                st.error(f"Task '{task_key_proc}': Selected features {missing_feats_proc} not found in source data from {test_data_path_proc}. Skipping.")
                continue
            if target_variable_proc not in df_source_for_task.columns:
                st.error(f"Task '{task_key_proc}': Target variable '{target_variable_proc}' not found in source data from {test_data_path_proc}. Skipping.")
                continue
            
            # Ensure there are features to select after all checks
            if not selected_features_proc:
                st.error(f"Task '{task_key_proc}': No features selected or available. Skipping.")
                continue


            df_with_attrs = df_source_for_task.copy()
            df_with_attrs.attrs["name"] = task_info_proc.get("task_name", task_key_proc)
            df_with_attrs.attrs["target_variable"] = target_variable_proc
            df_with_attrs.attrs["selected_features"] = selected_features_proc
            df_with_attrs.attrs["model_name"] = model_name_proc
            df_with_attrs.attrs["hyperparameters"] = hyperparams_proc
            df_with_attrs.attrs["configuration_source"] = source_of_config
            df_with_attrs.attrs["original_dev_metrics"] = task_info_proc.get("key_metrics", {})


            X_data = df_source_for_task[selected_features_proc]
            y_data = df_source_for_task[target_variable_proc]

            model_instance_proc = create_model_instance(model_name_proc, hyperparams_proc)
            if model_instance_proc is None:
                st.warning(f"Skipping model processing for '{model_name_proc}' (task '{task_key_proc}') due to model instantiation failure.")
                continue
            
            # Split data from df_source_for_task for training and evaluation here
            X_train_here, X_test_here, y_train_here, y_test_here = train_test_split(
                X_data, y_data, test_size=0.25, random_state=42 # Using 25% for test here
            )
            
            # Handle potential label encoding for classification targets if data is not already numeric
            # This is important if using raw default data that wasn't processed by Model_develop.py
            model_problem_type_proc = "Regression" if model_name_proc == "Linear Regression" else "Classification"
            y_train_fit = y_train_here.copy()
            y_test_eval = y_test_here.copy()
            label_encoder_map = None

            if model_problem_type_proc == "Classification":
                # Check if target is already numeric (0,1) or needs encoding
                if not pd.api.types.is_numeric_dtype(y_train_fit) or y_train_fit.nunique() > 20: # Heuristic for categorical
                    le = LabelEncoder()
                    y_train_fit = le.fit_transform(y_train_fit)
                    y_test_eval = le.transform(y_test_eval)
                    if hasattr(le, 'classes_'):
                        label_encoder_map = {i: str(cls_item) for i, cls_item in enumerate(le.classes_)} # Store mapping
                        df_with_attrs.attrs["label_encoding_applied_map"] = label_encoder_map


            model_instance_proc.fit(X_train_here, y_train_fit)
            
            # Predictions on the *entire* X_data from df_source_for_task
            y_pred_full_task = model_instance_proc.predict(X_data)
            df_with_attrs['Predicted_Target'] = y_pred_full_task

            # If label encoding was applied, try to map predictions back to original labels for a new column
            if label_encoder_map:
                try:
                    # Ensure y_pred_full_task contains values that are keys in label_encoder_map
                    valid_pred_keys = [p for p in y_pred_full_task if p in label_encoder_map]
                    original_labels_pred = pd.Series(y_pred_full_task).map(label_encoder_map)
                    df_with_attrs['Predicted_Target_Original_Label'] = original_labels_pred.fillna(y_pred_full_task) # Fallback for unmapped
                except Exception as e_map_back:
                    # st.warning(f"Could not map predictions back to original labels for {task_key_proc}: {e_map_back}")
                    df_with_attrs['Predicted_Target_Original_Label'] = y_pred_full_task # Store as is

            # Evaluation Metrics (on the X_test_here, y_test_eval split)
            calculated_metrics = {}
            y_pred_on_test_split_here = model_instance_proc.predict(X_test_here)

            if model_problem_type_proc == "Regression":
                calculated_metrics["R-squared"] = r2_score(y_test_eval, y_pred_on_test_split_here)
                calculated_metrics["MSE"] = mean_squared_error(y_test_eval, y_pred_on_test_split_here)
                calculated_metrics["RMSE"] = np.sqrt(calculated_metrics["MSE"])
            elif model_problem_type_proc == "Classification":
                pos_label_val = 1 # Default for binary after encoding
                avg_strat = 'binary' if len(np.unique(y_test_eval)) <= 2 else 'weighted'

                try:
                    y_proba_on_test_split_here = model_instance_proc.predict_proba(X_test_here)[:, 1]
                    calculated_metrics["AUC"] = roc_auc_score(y_test_eval, y_proba_on_test_split_here)
                except (AttributeError, ValueError) : # Model might not have predict_proba or y_test_eval might not be suitable for AUC (e.g. single class)
                    calculated_metrics["AUC"] = None 
                
                calculated_metrics["Accuracy"] = accuracy_score(y_test_eval, y_pred_on_test_split_here)
                calculated_metrics["Precision"] = precision_score(y_test_eval, y_pred_on_test_split_here, zero_division=0, pos_label=pos_label_val if avg_strat == 'binary' else None, average=avg_strat)
                calculated_metrics["Recall"] = recall_score(y_test_eval, y_pred_on_test_split_here, zero_division=0, pos_label=pos_label_val if avg_strat == 'binary' else None, average=avg_strat)
                calculated_metrics["F1 Score"] = f1_score(y_test_eval, y_pred_on_test_split_here, zero_division=0, pos_label=pos_label_val if avg_strat == 'binary' else None, average=avg_strat)
            
            df_with_attrs.attrs["evaluation_metrics_on_backtest_split"] = calculated_metrics
            processed_dataframes_list.append(df_with_attrs)

        except Exception as e_task_proc:
            st.error(f"An critical error occurred while processing task '{task_key_proc}' in back.py: {e_task_proc}")
            # st.exception(e_task_proc) # Uncomment for full traceback if needed
            
    return processed_dataframes_list


# --- Main Streamlit Page Logic (Displays Confirmation/Key Results) ---
def main_backtesting_page():
    st.header("Step 3: Backtesting Setup & Confirmation")

    confirmed_models_to_display = st.session_state.get("confirmed_model_outputs")
    is_using_defaults_for_display = False

    if not confirmed_models_to_display:
        st.info("No models were confirmed in the 'Model Development' step. Displaying default model configurations for demonstration.")
        confirmed_models_to_display = generate_default_confirmed_outputs()
        is_using_defaults_for_display = True
        if not confirmed_models_to_display: # If default generation also failed
            st.error("Failed to generate default model configurations. Please ensure 'default_on_us_data.parquet' is available and contains necessary columns, or confirm models in the 'Model Development' page.")
            return 
    
    if is_using_defaults_for_display:
        st.markdown("> :information_source: **You are viewing default model configurations.** Metrics displayed under 'Key Metrics (from Development Step)' are placeholders. Actual evaluation for these defaults will occur if you proceed to Model Monitoring, using a new train/test split of the `default_on_us_data.parquet` file.")
    else:
        st.info("These are the models and iterations you selected in the 'Model Development' step. Their test data slices and configurations are listed below.")
    
    st.markdown("Clicking 'Prepare Data and Proceed...' will finalize these configurations for the Model Monitoring page. This involves re-training each model on a portion of its specified data and generating predictions on the entirety of that data.")


    if not confirmed_models_to_display:
        st.warning("No model configurations available to display (neither user-confirmed nor default).")
        return
    
    st.success(f"Ready to display {len(confirmed_models_to_display)} model task configurations {'(Defaults)' if is_using_defaults_for_display else '(User-Confirmed)'}.")

    for task_key_disp, task_info_disp in confirmed_models_to_display.items():
        st.markdown(f"---")
        st.markdown(f"#### Task: {task_info_disp.get('task_name', task_key_disp)}")
        try:
            target_disp = task_info_disp.get("target_variable")
            model_name_disp = task_info_disp.get("model_name")
            hyperparams_disp = task_info_disp.get("hyperparameters")
            dev_metrics_disp = task_info_disp.get("key_metrics", {}) 
            data_path_disp = task_info_disp.get("test_data_path") # This is the source data for this task
            features_disp = task_info_disp.get("selected_features", [])
            iter_num_disp = task_info_disp.get("iteration_number", "N/A (Default)")

            st.markdown(f"**Target Variable:** `{target_disp}` | **Model:** `{model_name_disp}` | **Source Iteration:** `{iter_num_disp}`")
            st.markdown(f"**Data Source for this task:** `{data_path_disp}`")
            
            with st.expander("View Selected Features & Hyperparameters"):
                st.markdown(f"**Selected Features ({len(features_disp)}):**")
                st.json(features_disp) 
                st.markdown(f"**Hyperparameters:**")
                st.json(hyperparams_disp)

            st.markdown(f"**Key Metrics (from Development Step Test Set {'- N/A for Defaults' if is_using_defaults_for_display and dev_metrics_disp.get('Info') else ''}):**")
            if dev_metrics_disp and not dev_metrics_disp.get("Info"): # Check if not placeholder
                 problem_type_for_metrics = "Regression" if target_disp == "Profitability_GBP" else "Classification"
                 num_cols_for_metrics = 3 if problem_type_for_metrics == "Regression" else 5
                 metric_cols_on_page = st.columns(num_cols_for_metrics)
                 
                 metrics_order_display = []
                 if problem_type_for_metrics == "Classification":
                      metrics_order_display = ["ROC-AUC", "F1 Score", "Accuracy", "Precision", "Recall"]
                 elif problem_type_for_metrics == "Regression":
                      metrics_order_display = ["Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)", "R-squared"]

                 metric_idx_disp = 0
                 for metric_name_on_page in metrics_order_display:
                      val_on_page = dev_metrics_disp.get(metric_name_on_page)
                      if val_on_page is not None:
                           val_str_on_page = f"{val_on_page:.4f}" if isinstance(val_on_page, (int, float, np.float64)) else str(val_on_page)
                           metric_cols_on_page[metric_idx_disp % num_cols_for_metrics].metric(metric_name_on_page, val_str_on_page)
                           metric_idx_disp +=1
                 if metric_idx_disp == 0:
                     st.caption("Metrics from development step either not applicable or not found in the standard list.")
                     st.json(dev_metrics_disp) # Show raw if no standard metrics found
            elif is_using_defaults_for_display:
                st.caption("Metrics from the 'Model Development' step are not applicable for default configurations. New metrics will be calculated upon proceeding.")
            else:
                 st.caption("No key metrics were explicitly passed from the Model Development step for this iteration, or they were placeholders.")

        except Exception as e_main_page_disp:
            st.error(f"An error occurred while displaying configuration details for task {task_key_disp}: {e_main_page_disp}")

    st.markdown("---")
    if st.button("Prepare Data and Proceed to Model Prediction Monitoring (Next Page)", key="proceed_to_model_monitoring_page"):
         st.info("Preparation for Model Monitoring initiated. When you navigate to the 'Model Monitoring' page, it will use the configurations processed by this step.")
         st.session_state["backtesting_setup_complete_flag"] = True # Flag for next page if needed
         st.success("Setup complete! You can now navigate to 'Model Monitoring' via the sidebar.")
         # If using st.navigation or st.switch_page:
         # st.switch_page("pages/model_monitoring.py") # Example navigation


# --- Entry point for the Streamlit page ---
if __name__ == "__main__":
    main_backtesting_page()
