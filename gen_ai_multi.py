import os
import google.generativeai as genai
import streamlit as st # Assuming Streamlit secrets might be used for API key

class GenAIAgent:
    """
    A class to encapsulate interactions with the Google Gemini Generative AI model
    for generating Python code snippets for feature transformations.
    """
    def __init__(self):
        self._configure_gemini_api()
        self.model = self._initialize_generative_model()

    def _configure_gemini_api(self):
        """
        Configures the Gemini API using an API key from Streamlit secrets or environment variables.
        """
        # Access API key securely
        GEMINI_API_KEY = st.secrets["AIzaSyBTQyDIFyoeLBSdt-ttd14Oynd3zmxAl70"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            # In a production environment, you might want to raise an exception here
            # or log a critical error instead of just a Streamlit error.
            st.error("Gemini API Key not found. Please set it in Streamlit secrets or as an environment variable.")
            raise ValueError("GEMINI_API_KEY not configured.")

        genai.configure(api_key=GEMINI_API_KEY)
        st.success("Gemini API configured successfully.") # For debugging during setup

    def _initialize_generative_model(self):
        """
        Initializes and returns the Generative Model instance.
        """
        return genai.GenerativeModel(
            'gemini-pro', # Or 'gemini-1.5-pro-latest' if you have access and want more power
            generation_config={
                "temperature": 0.8, # Lower temperature for more deterministic, factual responses (code generation)
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 500, # Limit output length to prevent overly long code
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

        Args:
            features: A list of column names (e.g., ['col_A', 'col_B']).
            user_operation_text: The natural language description of the transformation.

        Returns:
            A string containing the generated Python code snippet (e.g., "df['col_A'] / df['col_B']").
            Returns an error message string if the LLM cannot generate valid code, prefixed with 'ERROR: '.
        """
        if not self.model: # Check if model was initialized (e.g. if API key was missing)
            return "ERROR: Gemini model not initialized due to API key issue."

        # Create a string representation of features for the prompt
        features_str = ", ".join([f"df['{f}']" for f in features])
        if len(features) == 1:
            features_context = f"The selected feature is: {features_str}. "
        elif len(features) > 1:
            features_context = f"The selected features are: {features_str}. "
        else:
            return "ERROR: No features provided for transformation."

        # --- LLM System Prompt ---
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

        # --- User Prompt ---
        user_prompt = (
            f"Features available for operation: {features_str}\n"
            f"User requested operation: '{user_operation_text}'\n"
            "Generate the pandas code for this transformation:"
        )

        try:
            response = self.model.generate_content([system_prompt, user_prompt])
            generated_text = response.text.strip()

            # Robust validation: ensure it's not empty, doesn't contain markdown or common code structures
            if not generated_text or "```" in generated_text or generated_text.lower().startswith(("import", "def", "class", "if", "for", "while")):
                return "ERROR: Invalid code generated by AI. The output should be a single pandas expression."
            if generated_text.startswith("ERROR:"): # Pass through LLM's own error messages
                return generated_text

            return generated_text
        except Exception as e:
            return f"ERROR: LLM API call failed: {str(e)}"

# Instantiate the agent once to reuse the model
gen_ai_agent = GenAIAgent()