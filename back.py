import json

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import os

from sklearn.linear_model import LogisticRegression, LinearRegression

from xgboost import XGBClassifier, XGBRegressor # If you have it installed

from lightgbm import LGBMClassifier, LGBMRegressor

import glob

import streamlit as st # Import streamlit


def create_model(model_name, hyperparameters):
    try:
        model_classes = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "XGBoost Classifier": XGBClassifier,
            "XGBoost Regressor": XGBRegressor,
            "LGBM Classifier": LGBMClassifier,
            "LGBM Regressor": LGBMRegressor
        }
        model_class = model_classes.get(model_name)
        if model_class:
            return model_class(**hyperparameters)
        else:
            print(f"Warning: Model name '{model_name}' not recognized. Skipping.")
            return None
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return None

def process_models_from_session():
    dataframes = []
    model_infos = st.session_state.get("selected_iterations", {})
    for model_key, model_info in model_infos.items():
        rows = model_info.get("dataset")
        if not rows:
            print(f"Error: 'dataset' key not found in model info for {model_key}")
            continue

        df = pd.DataFrame(rows)
        df['source_model'] = model_key

        target_variable = model_info.get("target_variable")
        model_name = model_info.get("model_name")
        hyperparameters = model_info.get("hyperparameters")
        model_code = model_info.get("model_code")

        if not all([target_variable, model_name, hyperparameters, model_code]):
            print(f"Error: Missing required info in model info for {model_key}")
            continue

        features = [col for col in df.columns if col != target_variable and col not in ['timestamp', 'date', 'loan_start_date', "TERM_OF_LOAN", "source_model"] and pd.api.types.is_numeric_dtype(df[col])]
        X = df[features]
        y = df[target_variable] if target_variable in df.columns else None

        model = create_model(model_name, hyperparameters)
        if model is None:
            print(f"Skipping model {model_key} due to invalid model.")
            continue

        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            print(f"Model {model_name} trained successfully for target variable {target_variable} from {model_key}")
            y_pred = model.predict(X)
            df['Predicted_Target'] = y_pred
            print(f"Evaluation Metrics for {model_name} - {target_variable} - {model_key}:")
            if target_variable == "Profitability_GBP":
                r2 = r2_score(y_test, y_pred[X_test.index])
                print(f"  R-squared: {r2:.4f}")
            else:
                try:
                    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    print(f"  AUC: {auc:.4f}")
                except Exception as e:
                    print(f"  AUC not calculated: {e}")
                acc = accuracy_score(y_test, y_pred[X_test.index])
                prec = precision_score(y_test, y_pred[X_test.index], zero_division=0)
                rec = recall_score(y_test, y_pred[X_test.index])
                f1 = f1_score(y_test, y_pred[X_test.index])
                print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        else:
            y_pred = model.predict(X)
            df['Predicted_Target'] = y_pred
            print(f"Trained model not evaluated for {target_variable} in {model_key}")

        df.attrs['name'] = f"{target_variable}_{model_name}"
        dataframes.append(df)

    return dataframes

def create_streamlit_app():
    st.title("DataFrames from Session State")
    processed_dataframes = process_models_from_session()
    if not processed_dataframes:
        st.error("Error processing models from session state. Please check the data.")
        return

    for df in processed_dataframes:
        st.header(f"DataFrame: {df.attrs.get('name', 'Unnamed DataFrame')}")
        st.dataframe(df.head())
        st.write(f"Columns: {', '.join(df.columns)}")

if __name__ == "__main__":
    create_streamlit_app()

