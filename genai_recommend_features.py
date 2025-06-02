# genai_utils.py
import ast 
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re
import numpy as np
import streamlit as st
from typing import List, Dict, Any
import google.generativeai as genai
import time
# Load environment variables and configure Gemini API
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_CONFIGURED = False
GEMINI_MODELS = {
    'gemini-2.0-flash': {
        'model': 'gemini-2.0-flash',
        'temperature': 0.1,
        'max_retries': 3,
        'chunk_size': 12000,
        'chunk_overlap': 1200
    },
    'gemini-1.5-flash': {
        'model': 'gemini-1.5-flash',
        'temperature': 0.1,
        'max_retries': 3,
        'chunk_size': 10000,
        'chunk_overlap': 1000
    },
    'gemini-1.5-pro': {
        'model': 'gemini-1.5-pro',
        'temperature': 0.1,
        'max_retries': 3,
        'chunk_size': 15000,
        'chunk_overlap': 1500
    },
    'gemini-pro': {
        'model': 'gemini-pro',
        'temperature': 0.1,
        'max_retries': 3,
        'chunk_size': 12000,
        'chunk_overlap': 1200
    }
}
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_API_KEY_CONFIGURED = True
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY_CONFIGURED = False
else:
    print("GEMINI_API_KEY not found in environment variables. Please set it in your .env file or environment.")

def summarize_dataset_columns(df: pd.DataFrame) -> str:
    """Generates a textual description of the dataset based on column names, data types, and examples."""
    summary_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        example_values = df[col].dropna().unique()[:3]  # Show a few unique examples
        example_str = ", ".join(map(str, example_values))
        summary_lines.append(f"- {col} (type: {dtype}, examples: [{example_str}])")
    return "\n".join(summary_lines)

def get_recommended_features_gemini(dataset_description: str):
    """
    Calls the Gemini API to recommend engineered features for loan profitability prediction.
    """
    if not GEMINI_API_KEY_CONFIGURED:
        return "Error: Gemini API key not configured. Cannot get recommendations."
    if st.session_state.get("recommend_features_done", False):
        st.info("Feature recommendations have already been processed. Skipping LLM calls.")
        return None
    llm_prompt = (
        "You are an expert data science assistant specializing in financial risk modeling.\n\n"
        "I have a loan dataset intended for profitability prediction. The profitability is calculated by "
        "estimating the probabilities of two key events:\n"
        "1. Charge-Off (COF_EVENT_LABEL) – when a loan is written off.\n"
        "2. Prepayment (PREPAYMENT_EVENT_LABEL) – when a loan is paid off early.\n"
        "The combined probabilities from these events are used to compute the expected profitability for each loan.\n\n"

        "Here is a summary of the available features in my raw dataset:\n"
        f"{dataset_description}\n\n"
        
        "Objective:\n"
        "My goal is to use machine learning models (Logistic Regression, LightGBM, XGBoost, Random Forest Classifiers) "
        "to predict these events. Based on the provided raw dataset features, I need you to perform **feature engineering** "
        "and recommend a ranked list of **new, engineered features** that would significantly enhance the predictive power "
        "of my models for Charge-Off, Prepayment, and overall loan profitability.\n\n"
        "Instructions:\n"
        "1. Always use the provided raw dataset features as the basis for your recommendations. Do not assume any external features.\n"
        "2. Identify Feature Engineering Opportunities from the features.\n"
        "3. Suggest only high-value, business-relevant engineered features.\n"
        "4. For each engineered feature, clearly state its name, how it's derived, the justification for its inclusion, "
        "and whether it's most relevant for Charge-Off, Prepayment, or Both.\n"
        "5. Provide a Ranked List in this exact format for each feature (ensure to strictly follow this multi-line format for each feature):\n"
        "- Engineered Feature Name: [Name]\n"
        "- Description: [Brief description of the feature]\n"
        "- List of Raw Features Used: [List of raw features used to create this engineered feature]\n"
        "- Derivation: [Explanation of how to compute the feature]\n"
        "- Code Snippet: [Python code snippet to compute the feature], give only the pandas operation code without defining extra functions ,my pandas df is df - just 1 line per feature put code inside [[[code]]] ,everything except code to be removed (justification,primary impact, etc..)\n"
        "- Justification: [Why this feature is valuable]\n"
        "- Primary Event Impact: [Charge-Off / Prepayment / Both]\n\n"
        "Begin the list directly. Do not include any preamble before the first feature. Ensure each feature block is separated by a double newline if providing multiple features."
    )

    for model_name, config in GEMINI_MODELS.items():
        genai_model = genai.GenerativeModel(config['model'])
        response = genai_model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config['temperature']
            )
        )
        return response.text
            

    
    
def get_code_snippet_gemini(code_snippet: str):
    """
    Calls the Gemini API to recommend engineered features for loan profitability prediction.
    """
    if not GEMINI_API_KEY_CONFIGURED:
        return "Error: Gemini API key not configured. Cannot get recommendations."

    llm_prompt = '''You will be given a section labeled as a "Code Snippet" that may contain non-code text or formatting artifacts. Extract and return only the valid, executable Python code found within it. The output must be in the following JSON format:
    {"code": "<cleaned_python_code>"}
    
    Rules:

    1.Only include actual Python code that can be executed with exec().
    2.Strip away any formatting markers (e.g., [[[code]]], [[[/code]]], markdown formatting, or non-code descriptions).
    3.Ensure the returned code is syntactically correct.
    4.Do not include explanations, comments, or additional text — only the JSON object containing the clean Python code.

    Example Input:

    [[[code]]]df['NEW_FEATURE'] = df['COL1'] / df['COL2'][[[/code]]]

    Expected Output:

    {"code": "df['NEW_FEATURE'] = df['COL1'] / df['COL2']"}

    Now process the following input:
    

    ''' + code_snippet
        
    for model_name, config in GEMINI_MODELS.items():
        genai_model = genai.GenerativeModel(config['model'])
        response = genai_model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config['temperature']
            )
        )
        return response.text
           

def parse_gemini_recommendations(text: str) -> pd.DataFrame:
    """
    Parses the text output from Gemini into a DataFrame.
    It expects features to be separated by double newlines or be identifiable by '- Engineered Feature Name:'.
    """
    features_list = []
    
    # Attempt to split by double newlines first, as it's a common separator for distinct blocks.
    # Then, refine by looking for the specific feature name marker.
    # Using a regex to split by one or more empty lines, then filtering out empty strings.
    raw_feature_blocks = [block for block in re.split(r'\n\s*\n', text.strip()) if block.strip()]

    if not raw_feature_blocks or not raw_feature_blocks[0].strip().startswith("- Engineered Feature Name:"):
        # If the above split doesn't work well, fallback to splitting by the feature name itself.
        # This helps if the LLM doesn't use double newlines consistently.
        # Positive lookahead `(?=...)` keeps the delimiter as part of the next split.
        raw_feature_blocks = [block for block in re.split(r'(?=- Engineered Feature Name:)', text.strip()) if block.strip()]


    for block_idx, block_content in enumerate(raw_feature_blocks):
        block = block_content.strip()
        if not block:
            continue

        feature_data = {}
        try:
            name_match = re.search(r"- Engineered Feature Name:\s*(.*)", block)
            derivation_match = re.search(r"- Derivation:\s*(.*)", block, re.DOTALL)
            justification_match = re.search(r"- Justification:\s*(.*)", block, re.DOTALL)
            impact_match = re.search(r"- Primary Event Impact:\s*(.*)", block)
            description_match = re.search(r"- Description:\s*(.*)", block, re.DOTALL)
            raw_data_match = re.search(r"- List of Raw Features Used:\s*(.*)", block, re.DOTALL)
            code_snippet_match = re.search(r"- Code Snippet:\s*(.*)", block, re.DOTALL)

            if name_match:
                feature_data['Feature'] = name_match.group(1).strip()
            else:
                # print(f"Skipping block (index {block_idx}), couldn't find Feature Name: '{block[:100]}...'")
                continue # Essential field missing

            def clean_multiline_field(text_block_content, field_text_match_group):
                if not field_text_match_group: return "N/A"
                # Split the field's text by lines
                lines = field_text_match_group.strip().split('\n')
                cleaned_lines = []
                for line in lines:
                    # Stop if we hit the start of another known field for the *current* feature block
                    if line.strip().startswith("- Derivation:") and "Derivation" not in feature_data or \
                       line.strip().startswith("- Justification:") and "Justification" not in feature_data or \
                       line.strip().startswith("- Primary Event Impact:") and "Primary Event Impact" not in feature_data or \
                       line.strip().startswith("- List of Raw Features Used:") and "Raw Features" not in feature_data or \
                       line.strip().startswith("- Code Snippet:") and "Code Snippet" not in feature_data :
                        if field_text_match_group.strip() != line.strip(): # make sure it's not the current line itself
                             break
                    cleaned_lines.append(line.strip())
                return " ".join(cleaned_lines).strip()

            derivation_text = derivation_match.group(1) if derivation_match else "N/A"
            justification_text = justification_match.group(1) if justification_match else "N/A"
            description_text = description_match.group(1) if description_match else "N/A"
            feature_data['Derivation'] = clean_multiline_field(block, derivation_text)
            feature_data['Justification'] = clean_multiline_field(block, justification_text)
            feature_data['Primary Event Impact'] = impact_match.group(1).strip() if impact_match else "N/A"
            feature_data['Raw Features'] = clean_multiline_field(block, raw_data_match.group(1) if raw_data_match else "N/A")
            feature_data['Code Snippet'] = clean_multiline_field(block, code_snippet_match.group(1) if code_snippet_match else "N/A")
            feature_data['Description'] = clean_multiline_field(block, description_text)
            
            feature_data['Min'] = 'N/A'
            feature_data['Max'] = 'N/A'
            feature_data['Mean'] = 'N/A'
            feature_data['Data Type'] = 'Engineered (Proposed)'
            
            features_list.append(feature_data)
             # Debugging point to inspect feature_data
        except Exception as e:
            print(f"Error parsing a feature block (index {block_idx}): {e}. Block content: '{block[:100]}...'")
            continue
            
    if not features_list:
        # This case will be handled in the Streamlit app, but good to be aware of
        pass
    # print(pd.DataFrame(features_list))
    return pd.DataFrame(features_list)



import json

import re

def extract_code_block(text: str) -> str:
    """
    Extracts the Python code snippet from a non-JSON string formatted like:
    ```json
    {"code": "<code_here>"}
    ```
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input is not a valid string.")
    
    # Use regex to find the code within {"code": "..."} pattern
    match = re.search(r'{"code":\s*"(.+?)"}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Code block not found or improperly formatted.")






def apply_recommended_features(current_dataset: pd.DataFrame, recommended_features: any) -> pd.DataFrame:
    """
    Applies recommended features to the current dataset by executing dynamically generated code snippets.
    """
    if isinstance(recommended_features, list):
        recommended_features = pd.DataFrame(recommended_features)

    if not isinstance(recommended_features, pd.DataFrame):
        raise ValueError("Recommended features must be a DataFrame or a list of dictionaries.")

    if "Code Snippet" not in recommended_features.columns:
        raise ValueError("Recommended features DataFrame must contain a 'Code Snippet' column.")

    updated_dataset = current_dataset.copy()

    output_dict = {'res_error':False,
                   'error':''}
    try:
        for feature_info in recommended_features.to_dict(orient="records"):
            feature_name = feature_info.get("Feature")
            generated_code = feature_info.get("Code Snippet")
            generated_code = get_code_snippet_gemini(generated_code)
            print("generated_code:")
            print(generated_code)
            sanitized_code = extract_code_block(str(generated_code))
            print("sanitized_code")
            print(sanitized_code)

            if not feature_name or not generated_code:
                st.warning(f"Invalid feature info: {feature_info}")
                continue

            # try:
            # Prepare execution context with a copy of the current dataset, numpy and pandas
            execution_context = {'df': updated_dataset.copy(), 'np': np, 'pd': pd}

            # Execute the sanitized code snippet
            exec(str(sanitized_code), {}, execution_context)
            
            # Get the possibly modified DataFrame
            new_df = execution_context.get("df")

            # Defensive: check if new_df is valid
            if new_df is None or not isinstance(new_df, pd.DataFrame):
                raise ValueError("Executed code did not return a valid DataFrame.")

            # Check if the feature column exists exactly as expected
            if feature_name in new_df.columns:
                updated_dataset = new_df
                continue

            # Fallback 1: Look for variable in context with underscores replacing spaces and lowercase
            feature_var_name = feature_name.replace(" ", "_").lower()

            matched_var = None
            for var_name, var_value in execution_context.items():
                if var_name.lower() == feature_var_name:
                    matched_var = var_value
                    break

            if matched_var is not None:
                updated_dataset[feature_name] = matched_var
                continue

            # Fallback 2: Check for any new columns added by the code snippet compared to updated_dataset
            new_columns = set(new_df.columns) - set(updated_dataset.columns)
            if new_columns:
                # Take the first new column and assign it to feature_name
                first_new_col = list(new_columns)[0]
                updated_dataset[feature_name] = new_df[first_new_col]
                continue
            
            # If none of the above worked, warn and skip
            st.warning(f"Feature '{feature_name}' not found or created after executing code snippet. Skipping.")

            # except Exception as e:
            #     st.error(f"❌ Failed to create feature '{feature_name}': {e}")
    
    except Exception as e:
        output_dict['res_error'] = True
        output_dict['error'] = e
    
    return updated_dataset,output_dict



    

def validate_code_snippet(code: str) -> bool:
    """
    Validates the generated code snippet to ensure it is valid Python code.
    Args:
        code (str): The code snippet to validate.
    Returns:
        bool: True if the code is valid, False otherwise.
    """
    try:
        ast.parse(code)  # Parse the code to check for syntax errors
        return True
    except SyntaxError as e:
        st.error(f"Syntax error in generated code: {e}")
        return False

def sanitize_code_snippet(code: str) -> str:
    """
    Extracts and sanitizes the generated code snippet to fix common issues like unterminated string literals.
    """
    try:
        # Remove surrounding triple quotes if they exist
        code = code.strip()
        if (code.startswith("'''") and code.endswith("'''")) or (code.startswith('"""') and code.endswith('"""')):
            code = code[3:-3].strip()
        return code
    except Exception as e:
        st.error(f"Error sanitizing code snippet: {e}")
        return code

    
def _dedent_code(code_str):
    """
    Remove common leading whitespace from each line in code_str.
    """
    lines = code_str.strip().split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():  # Ignore empty lines
            min_indent = min(min_indent, len(line) - len(line.lstrip()))
    return "\n".join(line[min_indent:] if line.strip() else "" for line in lines)