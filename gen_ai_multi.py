# gen_ai_multi.py

import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np

class GenAIAgent:
    """
    A class to encapsulate interactions with the Google Gemini Generative AI model
    for generating Python code snippets for feature transformations.
    """
    def __init__(self):
        self.model = None
        try:
            self._configure_gemini_api()
            # Only initialize model if API key was successfully found
            if genai.api_key: # Check if genai.api_key is set after configuration
                self.model = self._initialize_generative_model()
            else:
                st.warning("Gemini API key not found. AI features will be unavailable.")
        except Exception as e:
            st.error(f"Failed to initialize GenAIAgent: {e}")
            self.model = None # Ensure model is None if initialization fails

    def _configure_gemini_api(self):
        """
        Configures the Gemini API using an API key from Streamlit secrets or environment variables.
        """
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            st.error("Gemini API Key not found. Please set it in Streamlit secrets or as an environment variable.")
            # Do NOT raise ValueError here, as it prevents Streamlit from running.
            # Instead, ensure `self.model` remains None and subsequent calls check for it.
            return # Exit if key not found

        genai.configure(api_key=GEMINI_API_KEY)
        # st.success("Gemini API configured successfully.") # Good for debugging during setup, remove for production

    def _initialize_generative_model(self):
        # ... (rest of _initialize_generative_model method - NO CHANGE) ...
        return genai.GenerativeModel(
            'gemini-pro',
            generation_config={
                "temperature": 0.2,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 500,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )

    def generate_transform_code(self, features: list[str], user_operation_text: str) -> str:
        """
        Generates a Python code snippet using the Gemini LLM for feature transformation.
        """
        if not self.model: # Check if model was successfully initialized
            return "ERROR: Gemini model not initialized due to API key or configuration issue."

        # ... (rest of generate_transform_code method - NO CHANGE) ...
        if not features:
            return "ERROR: No features provided for transformation."

        features_str = ", ".join([f"df['{f}']" for f in features])
        features_context = f"The selected features are: {features_str}. "

        system_prompt = (
            "You are an expert Python programmer specializing in Pandas DataFrame manipulations for feature engineering. "
            "Your task is to generate a single line of Python code that calculates a new pandas Series. "
            "This code will be executed within a Python environment where 'df' is the current pandas DataFrame "
            "and 'np' refers to the numpy library. "
            "Your response MUST contain ONLY the Python code and nothing else. "
            "Do NOT include any explanations, comments, imports, or other text. "
            "If the operation is ambiguous or cannot be performed on the given features (e.g., non-numeric for math ops), "
            "return a specific error string starting with 'ERROR: ' and describe the problem clearly."
            "\n\n"
            "Examples:\n"
            "Input: Features: df['col_A'], df['col_B']. Operation: 'sum of col_A and col_B'\nOutput: df['col_A'] + df['col_B']\n"
            "Input: Features: df['col_X'], df['col_Y']. Operation: 'ratio of col_X to col_Y'\nOutput: np.where(df['col_Y'] != 0, df['col_X'] / df['col_Y'], np.nan)\n"
            "Input: Features: df['age']. Operation: 'square root of age'\nOutput: np.sqrt(df['age'])\n"
            "Input: Features: df['sales']. Operation: 'log transform of sales'\nOutput: np.log1p(df['sales'])\n"
            "Input: Features: df['price'], df['quantity']. Operation: 'product of price and quantity'\nOutput: df['price'] * df['quantity']\n"
            "Input: Features: df['date_column']. Operation: 'difference in days from 2023-01-01'\nOutput: (df['date_column'] - pd.to_datetime('2023-01-01')).dt.days\n"
            "Input: Features: df['text_column']. Operation: 'length of text_column'\nOutput: df['text_column'].str.len()\n"
            "Input: Features: df['col_A'], df['col_B']. Operation: 'average of col_A and col_B'\nOutput: (df['col_A'] + df['col_B']) / 2\n"
            "Input: Features: df['num_col'], df['cat_col']. Operation: 'average of num_col and cat_col'\nOutput: ERROR: Cannot average numeric and categorical columns. Please select only numeric features.\n"
        )

        user_prompt = (
            f"Features available for operation: {features_str}\n"
            f"User requested operation: '{user_operation_text}'\n"
            "Generate the pandas code for this transformation:"
        )

        try:
            response = self.model.generate_content([system_prompt, user_prompt])
            generated_text = response.text.strip()

            if not generated_text or "```" in generated_text or generated_text.lower().startswith(("import", "def", "class", "if", "for", "while")):
                return "ERROR: Invalid code generated by AI. The output should be a single pandas expression."
            if generated_text.startswith("ERROR:"):
                return generated_text

            return generated_text
        except Exception as e:
            return f"ERROR: LLM API call failed: {str(e)}"

# Instantiate the agent once to reuse the model
gen_ai_agent = GenAIAgent()