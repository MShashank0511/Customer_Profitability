import numpy as np
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
    roc_auc_score, confusion_matrix, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
import shap

st.set_page_config(page_title="Model Development", layout="wide")
st.title("🔧 Model Development")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded!")
    st.dataframe(df.head(), use_container_width=True)

    # Step 2: Sub-sampling
    st.subheader("🔍 Sub-sampling")
    sample_frac = st.slider("Select sub-sample fraction", 0.1, 1.0, 1.0)
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Step 3: Target Variable Selection
    st.subheader("🎯 Target Variable Selection")
    target_column = st.selectbox("Search and Select Target Variable", df.columns)

    if target_column:
        # Determine if the target variable is categorical or continuous
        if df[target_column].nunique() < 10:  # Arbitrary threshold for categorical
            target_type = "Categorical"
        else:
            target_type = "Continuous"

        st.markdown(f"**Target Variable Type:** {target_type}")

        if target_type == "Continuous":
            # Display histogram with KDE
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[target_column], kde=True, ax=ax, color="blue")
            ax.set_title(f"Distribution of {target_column}")
            ax.set_xlabel(target_column)
            st.pyplot(fig)
        else:
            # Display event rate as a card
            event_rate = (df[target_column].value_counts(normalize=True) * 100).to_dict()
            st.metric(
                label="Event Rate (%)",
                value=f"{event_rate.get(1, 0):.2f}%" if 1 in event_rate else "N/A",
                help="Percentage of positive events in the target variable."
            )

            # Display bar plot for frequency
            frequency_df = df[target_column].value_counts().reset_index()
            frequency_df.columns = [target_column, "count"]

            fig = px.bar(
                frequency_df,
                x=target_column,
                y="count",
                text="count",  # Add count as text inside the bars
                labels={target_column: target_column, "count": "Frequency"},
                title=f"Frequency of {target_column}"
            )

            # Update layout to display text inside the bars
            fig.update_traces(textposition="inside")  # Position text inside the bars
            st.plotly_chart(fig, use_container_width=True)

    # Step 4: Train/Test Split
    st.subheader("🧪 Train/Test Split")
    test_size = st.slider("Select test size", 0.1, 0.5, 0.2)

    # Step 5: Model Training
    st.subheader("⚙️ Model Training")
    feature_columns = st.multiselect("📊 Select Features", [col for col in df.columns if col != target_column])

    if feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        # Ensure y is a 1D array
        y = y.ravel()

        # Determine problem type
        if len(np.unique(y)) < 3:
            problem_type = "Classification"
        else:
            problem_type = "Regression"
        st.markdown(f"**Problem Type:** {problem_type}")

        # Encode target for classification
        if problem_type == "Classification" and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Step 6: Model Selection
        st.subheader("📚 Select Model")
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression,
                "LGBM Classifier": LGBMClassifier,
                "Random Forest Classifier": RandomForestClassifier
            }
        else:
            models = {"Linear Regression": LinearRegression}

        selected_model = st.selectbox("Select Model", list(models.keys()))
        model_class = models[selected_model]

        # Step 7: Run Model
        if st.button("Run Model"):
            st.subheader("🏆 Best 5 Iterations")
            param_dist = {
                "Logistic Regression": {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
                "LGBM Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
                "Random Forest Classifier": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]},
                "Linear Regression": {"fit_intercept": [True, False]}
            }

            sampled_params = list(ParameterSampler(param_dist[selected_model], n_iter=5, random_state=42))
            best_iterations = []

            for i, params in enumerate(sampled_params):
                try:
                    # Initialize the model with sampled hyperparameters
                    model = model_class(**params)

                    # Fit the model
                    model.fit(X_train, y_train)

                    # Predict
                    y_pred = model.predict(X_test)

                    # Metrics
                    metrics = {}
                    if problem_type == "Classification":
                        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                        metrics = {
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred, average='binary', zero_division=0),
                            "Recall": recall_score(y_test, y_pred, average='binary', zero_division=0),
                            "F1 Score": f1_score(y_test, y_pred, average='binary', zero_division=0),
                        }
                        if y_proba is not None:
                            metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])
                            metrics["Gini Index"] = 2 * metrics["ROC-AUC"] - 1

                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        metrics["Confusion Matrix"] = cm
                    else:
                        metrics = {
                            "Mean Squared Error": mean_squared_error(y_test, y_pred),
                            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
                        }

                    # Append results
                    best_iterations.append({"params": params, "metrics": metrics})

                except Exception as e:
                    st.warning(f"⚠️ Skipping iteration due to error: {e}")

            # Display Best Iterations
            for i, iteration in enumerate(best_iterations):
                with st.expander(f"Iteration {i+1}"):
                    st.markdown("### 🔧 Hyperparameters")
                    st.json(iteration["params"])

                    st.markdown("### 📊 Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{iteration['metrics'].get('Accuracy', 'N/A'):.4f}")
                    col2.metric("Precision", f"{iteration['metrics'].get('Precision', 'N/A'):.4f}")
                    col3.metric("Recall", f"{iteration['metrics'].get('Recall', 'N/A'):.4f}")
                    col4.metric("F1 Score", f"{iteration['metrics'].get('F1 Score', 'N/A'):.4f}")

                    if "ROC-AUC" in iteration["metrics"]:
                        col5, col6 = st.columns(2)
                        col5.metric("ROC-AUC", f"{iteration['metrics']['ROC-AUC']:.4f}")
                        col6.metric("Gini Index", f"{iteration['metrics']['Gini Index']:.4f}")

                    # Confusion Matrix and SHAP Plots
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if "Confusion Matrix" in iteration["metrics"]:
                            st.markdown("#### Confusion Matrix")
                            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                            sns.heatmap(iteration["metrics"]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                            ax_cm.set_xlabel("Predicted")
                            ax_cm.set_ylabel("Actual")
                            st.pyplot(fig_cm)

                    with col2:
                        st.markdown("#### SHAP Summary Plot")
                        try:
                            explainer = shap.Explainer(model, X_train)
                            shap_values = explainer(X_test)
                            shap.summary_plot(shap_values, X_test, show=False)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate SHAP summary plot: {e}")

                        st.markdown("#### SHAP Bar Plot (Top 5 Features)")
                        try:
                            shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=5, show=False)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate SHAP bar plot: {e}")

            # Step 8: Select Final Iteration
            st.subheader("✅ Select Final Iteration")
            selected_iteration = st.selectbox("Select Iteration", [f"Iteration {i+1}" for i in range(len(best_iterations))])
            if st.button("Pass Selected Iteration to Next Page"):
                st.success(f"Selected {selected_iteration} for the next page!")
