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

from sklearn.ensemble import RandomForestClassifier


def create_model(model_name, hyperparameters):
    model_classes = {
        "Linear Regression": LinearRegression,
        "Logistic Regression": LogisticRegression,
        "Random Forest Classifier": RandomForestClassifier,
        "LGBM Classifier": LGBMClassifier,
        "XGBoost Classifier": XGBClassifier
    }
    model_class = model_classes.get(model_name)
    if model_class:
        return model_class(**hyperparameters)
    return None

def process_models_from_session():
    dataframes = []
    model_infos = st.session_state.get("selected_iterations", {})
    for model_key, model_info in model_infos.items():
        df = pd.DataFrame(model_info["dataset"])
        df.attrs['name'] = f"{model_info['target_variable']}_{model_info['model_name']}"
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

