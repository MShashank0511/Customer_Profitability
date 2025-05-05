import streamlit as st
import pandas as pd
import numpy as np
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

    # Step 3: Train/Test Split
    st.subheader("🧪 Train/Test Split")
    test_size = st.slider("Select test size", 0.1, 0.5, 0.2)

    # Step 4: Model Training
    st.subheader("⚙️ Model Training")
    target_column = st.selectbox("🎯 Select Target Column", df.columns)
    feature_columns = st.multiselect("📊 Select Features", [col for col in df.columns if col != target_column])

    if feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        # Ensure y is a 1D array immediately after extraction
        if isinstance(y, pd.Series):
            y = y.values  # Convert Series to NumPy array
        elif isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].values  # Extract the first column if y is a DataFrame
        else:
            y = np.array(y)  # Convert to NumPy array if not already

        y = y.ravel()  # Flatten to ensure it's 1D

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

        # Ensure y_train and y_test are also 1D arrays
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        # Step 5: Model Selection
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

        # Step 6: Run Model
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

                    # Conditional handling for model-specific requirements
                    if problem_type == "Classification":
                        # For classification, encode the target variable if necessary
                        y_train_encoded = y_train.ravel()
                        y_test_encoded = y_test.ravel()
                    else:
                        # For regression, use the original target variable
                        y_train_encoded = y_train
                        y_test_encoded = y_test

                    # Fit the model
                    model.fit(X_train, y_train_encoded)

                    # Predict
                    y_pred = model.predict(X_test)

                    # Metrics
                    metrics = {}
                    if problem_type == "Classification":
                        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                        metrics = {
                            "Accuracy": accuracy_score(y_test_encoded, y_pred),
                            "Precision": precision_score(y_test_encoded, y_pred, average='binary', zero_division=0),
                            "Recall": recall_score(y_test_encoded, y_pred, average='binary', zero_division=0),
                            "F1 Score": f1_score(y_test_encoded, y_pred, average='binary', zero_division=0),
                        }
                        if y_proba is not None:
                            metrics["ROC-AUC"] = roc_auc_score(y_test_encoded, y_proba[:, 1])
                            metrics["Gini Index"] = 2 * metrics["ROC-AUC"] - 1

                        # Confusion Matrix
                        cm = confusion_matrix(y_test_encoded, y_pred)
                        metrics["Confusion Matrix"] = cm
                    else:
                        # For regression, calculate regression-specific metrics
                        metrics = {
                            "Mean Squared Error": mean_squared_error(y_test_encoded, y_pred),
                            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test_encoded, y_pred)),
                        }

                    # Append results
                    best_iterations.append({"params": params, "metrics": metrics})

                except Exception as e:
                    st.warning(f"⚠️ Skipping iteration due to error: {e}")

            # Display Best Iterations
            for i, iteration in enumerate(best_iterations):
                with st.expander(f"Iteration {i+1}"):
                    # Hyperparameters
                    st.markdown("### 🔧 Hyperparameters")
                    st.json(iteration["params"])

                    # Metrics
                    st.markdown("### 📊 Metrics")
                    if problem_type == "Classification":
                        # Align classification metrics horizontally
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{iteration['metrics']['Accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{iteration['metrics']['Precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{iteration['metrics']['Recall']:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{iteration['metrics']['F1 Score']:.4f}")

                        # Display additional metrics if available
                        if "ROC-AUC" in iteration["metrics"]:
                            col5, col6 = st.columns(2)
                            with col5:
                                st.metric("ROC-AUC", f"{iteration['metrics']['ROC-AUC']:.4f}")
                            with col6:
                                st.metric("Gini Index", f"{iteration['metrics']['Gini Index']:.4f}")

                    elif problem_type == "Regression":
                        # Display regression metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Squared Error", f"{iteration['metrics']['Mean Squared Error']:.4f}")
                        with col2:
                            st.metric("Root Mean Squared Error", f"{iteration['metrics']['Root Mean Squared Error']:.4f}")

                    # Feature Importance and Confusion Matrix
                    st.markdown("### 🔍 Feature Importance and Confusion Matrix")
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test)

                        # Create two columns for horizontal alignment
                        col1, col2 = st.columns(2)

                        # Confusion Matrix
                        with col1:
                            if "Confusion Matrix" in iteration["metrics"]:
                                st.markdown("#### Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots(figsize=(3, 3))  # Reduce the size of the confusion matrix
                                sns.heatmap(iteration["metrics"]["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                                ax_cm.set_xlabel("Predicted")
                                ax_cm.set_ylabel("Actual")
                                st.pyplot(fig_cm)  # Pass the figure to st.pyplot()

                        # SHAP Bar Plot for Top 5 Features
                        with col2:
                            st.markdown("#### Feature Importance (Top 5 Features)")
                            shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=5, show=False)  # Limit to top 5 features
                            fig_shap_bar = plt.gcf()  # Get the current figure
                            fig_shap_bar.set_size_inches(4, 3)  # Reduce the size of the bar plot
                            ax = fig_shap_bar.axes[0]  # Get the first axis of the figure
                            ax.set_xlabel("Avg Impact on Model Output", fontsize=10)
                            st.pyplot(fig_shap_bar)  # Pass the figure to st.pyplot()

                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP plots: {e}")

            # Step 7: Select Final Iteration
            st.subheader("✅ Select Final Iteration")
            selected_iteration = st.selectbox("Select Iteration", [f"Iteration {i+1}" for i in range(len(best_iterations))])
            if st.button("Pass Selected Iteration to Next Page"):
                st.success(f"Selected {selected_iteration} for the next page!")