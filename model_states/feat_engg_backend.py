import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st # Streamlit might be needed for st.warning/error in backend functions
import random # Required for random.sample in select_mandatory_features
from gen_ai_multi import gen_ai_agent


# --- Existing functions from Filtering Section ---

def get_table_names(datasets: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Returns a list of names of the available datasets.

    Args:
        datasets: A dictionary where keys are dataset names (str) and values are pandas DataFrames.

    Returns:
        A list of strings representing the names of the datasets.
    """
    return list(datasets.keys())


def get_features_for_table(dataset_name: str, datasets: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Returns a list of feature (column) names for a given dataset.

    Args:
        dataset_name: The name of the dataset.
        datasets: A dictionary where keys are dataset names (str) and values are pandas DataFrames.

    Returns:
        A list of strings representing the column names of the specified dataset,
        or an empty list if the dataset is not found or is empty.
    """
    if dataset_name in datasets and not datasets[dataset_name].empty:
        return datasets[dataset_name].columns.tolist()
    return []


def get_filter_operations() -> List[str]:
    """
    Returns a list of supported filtering operations.

    Returns:
        A list of strings representing the available filter operations.
    """
    # Based on common pandas filtering capabilities and user requests
    return [
        "Greater Than",
        "Less Than",
        "Equal To",
        "Not Equal To",
        "Greater Than or Equal To",
        "Less Than or Equal To",
        "Is In List",       # Check if value is in a list
        "Between",          # Check if value is within a range (inclusive)
        "Is Null",          # Check for missing values
        "Is Not Null",      # Check for non-missing values
        "Contains String"   # Check if a string column contains a substring
    ]


def _get_filter_mask(df: pd.DataFrame, filter_block: Dict[str, Any]) -> pd.Series:
    """
    Generates a boolean mask for a single filter block applied to a DataFrame.

    Args:
        df: The DataFrame to apply the filter to.
        filter_block: A dictionary defining the filter (feature, operation, value).
            Note: dataset and output_name from the block are not used here.

    Returns:
        A pandas Series of boolean values (True for rows to keep).

    Raises:
        ValueError: If the feature is not found or the operation/value is invalid.
        TypeError: If value type is incompatible with the operation or feature dtype.
        RuntimeError: For unexpected issues during mask generation.
    """
    feature_name = filter_block.get("feature")
    operation = filter_block.get("operation")
    value = filter_block.get("value")

    if not feature_name:
        raise ValueError("Filter block missing feature name.")
    # Check if feature_name exists in the current DataFrame columns
    # This check is also done before calling, but included here for robustness
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame columns.")
    if not operation:
         raise ValueError("Filter block missing operation.")

    col = df[feature_name]
    col_dtype = col.dtype

    # Handle different operations
    try: # Wrap operation logic in a try block to catch type/value errors more specifically
        if operation == "Greater Than":
            if not pd.api.types.is_numeric_dtype(col_dtype):
                raise TypeError(f"Feature '{feature_name}' is not numeric for 'Greater Than' operation.")
            if not isinstance(value, (int, float)):
                 raise TypeError(f"Value for 'Greater Than' on '{feature_name}' must be numeric.")
            mask = col > value
        elif operation == "Less Than":
             if not pd.api.types.is_numeric_dtype(col_dtype):
                 raise TypeError(f"Feature '{feature_name}' is not numeric for 'Less Than' operation.")
             if not isinstance(value, (int, float)):
                  raise TypeError(f"Value for 'Less Than' on '{feature_name}' must be numeric.")
             mask = col < value
        elif operation == "Equal To":
             # Comparison should handle NaNs properly (NaN == NaN is False)
             # For equality with NaN, use isnull()
             if pd.isna(value):
                  mask = col.isnull()
             else:
                  mask = col == value # Works for most comparable types
        elif operation == "Not Equal To":
             # Comparison should handle NaNs properly (NaN != NaN is True)
             # For inequality with NaN, use notnull()
             if pd.isna(value):
                  mask = col.notnull()
             else:
                  mask = col != value # Works for most comparable types
        elif operation == "Greater Than or Equal To":
             if not pd.api.types.is_numeric_dtype(col_dtype):
                 raise TypeError(f"Feature '{feature_name}' is not numeric for 'Greater Than or Equal To' operation.")
             if not isinstance(value, (int, float)):
                  raise TypeError(f"Value for 'Greater Than or Equal To' on '{feature_name}' must be numeric.")
             mask = col >= value
        elif operation == "Less Than or Equal To":
             if not pd.api.types.is_numeric_dtype(col_dtype):
                 raise TypeError(f"Feature '{feature_name}' is not numeric for 'Less Than or Equal To' operation.")
             if not isinstance(value, (int, float)):
                  raise TypeError(f"Value for 'Less Than or Equal To' on '{feature_name}' must be numeric.")
             mask = col <= value
        elif operation == "Is In List":
             if not isinstance(value, list):
                 # Handle the case where value is still a comma-separated string from UI
                 if isinstance(value, str):
                      value_list = [v.strip() for v in value.split(',') if v.strip()]
                 else:
                      raise TypeError(f"Value for 'Is In List' on '{feature_name}' must be a list or comma-separated string.")
             else:
                  value_list = value
             # Ensure value_list is iterable and contains comparable types
             if not isinstance(value_list, (list, tuple, set)):
                  raise TypeError(f"Invalid value list for 'Is In List' on '{feature_name}'.")
             mask = col.isin(value_list)
        elif operation == "Between":
             if not pd.api.types.is_numeric_dtype(col_dtype):
                 raise TypeError(f"Feature '{feature_name}' is not numeric for 'Between' operation.")
             if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(isinstance(v, (int, float)) for v in value):
                  raise TypeError(f"Value for 'Between' on '{feature_name}' must be a tuple or list of two numeric values.")
             # Ensure value[0] is less than or equal to value[1] for expected behavior
             val1, val2 = sorted(value)
             mask = col.between(val1, val2, inclusive='both') # inclusive='both' is default but good to be explicit
        elif operation == "Is Null":
             mask = col.isnull()
        elif operation == "Is Not Null":
             mask = col.notnull()
        elif operation == "Contains String":
            # Ensure value is a string for contains operation
            if value is None: value_str = "" # Treat None value as empty string for contains
            else: value_str = str(value)

            if not pd.api.types.is_string_dtype(col_dtype):
                 # Attempt to convert to string if not already
                 try:
                      col_str = col.astype(str)
                      mask = col_str.str.contains(value_str, na=False) # Treat nulls as not containing string
                 except Exception:
                      raise TypeError(f"Feature '{feature_name}' cannot be converted to string for 'Contains String' operation.")
            else:
                 # Ensure the column doesn't have mixed types that prevent .str accessor
                 if not hasattr(col, 'str'):
                      try:
                          col = col.astype(str) # Try converting if .str accessor is missing
                          mask = col.str.contains(value_str, na=False)
                      except Exception:
                          raise TypeError(f"Feature '{feature_name}' does not support string operations for 'Contains String'.")
                 else:
                    mask = col.str.contains(value_str, na=False) # Treat nulls as not containing string

            if not value_str: # If the substring to search for is empty
                 # Contains an empty string is technically always True, but usually not desired for filtering.
                 # Let's make it return all rows if the search string is empty.
                 mask = pd.Series(True, index=df.index)


        else:
            raise ValueError(f"Unsupported filter operation: {operation}")

    except (ValueError, TypeError, Exception) as e:
         # Re-raise with more context
         raise type(e)(f"Error in '{operation}' operation on feature '{feature_name}' with value '{value}': {e}")


    # Ensure the mask is a boolean Series with the same index as the input DataFrame
    if not isinstance(mask, pd.Series) or not pd.api.types.is_bool_dtype(mask.dtype) or not mask.index.equals(df.index):
         # This indicates an issue in the filter logic for the operation
         raise RuntimeError(f"Internal error generating valid boolean mask for operation '{operation}' on feature '{feature_name}'.")


    return mask


def apply_filter_block(df: pd.DataFrame, filter_block: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a single filter operation to a DataFrame and adds a boolean column
    indicating which rows satisfy the filter.

    Args:
        df: The input pandas DataFrame.
        filter_block: A dictionary defining the filter operation, containing:
                      - 'feature': The name of the column to filter on.
                      - 'operation': The type of filter operation (str).
                      - 'value': The value(s) to use for filtering (can be a single value, list, or tuple).
                      - 'output_name': The name for the new boolean column.

    Returns:
        The DataFrame with a new boolean column added.
        Returns the original DataFrame if the filter cannot be applied (e.g., invalid feature).
    """
    feature = filter_block.get("feature")
    operation = filter_block.get("operation")
    value = filter_block.get("value")
    output_name = filter_block.get("output_name")

    if not feature or not operation or not output_name or feature not in df.columns:
        print(f"Skipping invalid filter block: {filter_block}") # Log for debugging
        return df # Return original DataFrame if block is invalid

    try:
        # Apply the filter based on the operation
        if operation == "Greater Than":
            df[output_name] = df[feature] > value
        elif operation == "Less Than":
            df[output_name] = df[feature] < value
        elif operation == "Equal To":
            df[output_name] = df[feature] == value
        elif operation == "Not Equal To":
            df[output_name] != value
        elif operation == "Greater Than or Equal To":
            df[output_name] = df[feature] >= value
        elif operation == "Less Than or Equal To":
            df[output_name] = df[feature] <= value
        elif operation == "Is In List":
            # Ensure value is a list
            if isinstance(value, list):
                df[output_name] = df[feature].isin(value)
            else:
                print(f"Warning: 'Is In List' operation requires a list value. Skipping filter for feature '{feature}'.")
                df[output_name] = False # Mark all rows as False for this filter
        elif operation == "Between":
            # Ensure value is a tuple or list of two elements
            if isinstance(value, (tuple, list)) and len(value) == 2:
                df[output_name] = df[feature].between(value[0], value[1])
            else:
                print(f"Warning: 'Between' operation requires a tuple or list of two values. Skipping filter for feature '{feature}'.")
                df[output_name] = False # Mark all rows as False for this filter
        elif operation == "Is Null":
            df[output_name] = df[feature].isnull()
        elif operation == "Is Not Null":
            df[output_name] = df[feature].notnull()
        elif operation == "Contains String":
             # Check if the feature column is of object (string) dtype
            if pd.api.types.is_object_dtype(df[feature].dtype):
                # Use .str accessor and handle potential NaNs
                df[output_name] = df[feature].str.contains(str(value), na=False)
            else:
                print(f"Warning: 'Contains String' operation is only applicable to string columns. Skipping filter for feature '{feature}'.")
                df[output_name] = False # Mark all rows as False for this filter
        else:
            print(f"Warning: Unsupported operation '{operation}'. Skipping filter for feature '{feature}'.")
            df[output_name] = False # Mark all rows as False for unsupported ops

    except Exception as e:
        print(f"Error applying filter for feature '{feature}' with operation '{operation}': {e}")
        df[output_name] = False # Mark all rows as False on error

    return df


def apply_all_filters_for_table(original_df: pd.DataFrame, filter_blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Applies a list of filter blocks sequentially to a DataFrame.
    Each subsequent filter is applied to the result of the previous filter.

    Args:
        original_df: The initial DataFrame to apply filters to.
        filter_blocks: A list of dictionaries, where each dictionary defines a filter block.
            These blocks are assumed to be for the *same* original dataset,
            and will be applied in the order they appear in the list.

    Returns:
        A new DataFrame containing only the rows that satisfy all filters sequentially applied,
        or the original DataFrame if filter_blocks is empty.
    """
    if not filter_blocks:
        return original_df.copy() # Return a copy if no filters

    current_df = original_df.copy()
    applied_filters_count = 0

    for i, block in enumerate(filter_blocks):
        # The block contains dataset and feature names from the original context.
        # We need to apply this filter logic to the current_df using the feature name.

        feature_name = block.get("feature")
        operation = block.get("operation")
        value = block.get("value")
        # dataset_name = block.get("dataset") # Original dataset name from the block, not used for application on current_df
        # output_name = block.get("output_name") # Output name for the boolean column, not used in sequential filtering result

        # Skip incomplete blocks
        if not feature_name or not operation:
             continue # Skip this block if it's incomplete


        # Check for feature existence in current_df columns BEFORE getting the mask
        if feature_name not in current_df.columns:
             st.warning(f"Skipping filter block {i+1} ('{feature_name}' {operation} {value}) - Feature not found in current intermediate data.")
             continue # Skip to the next filter block if feature is missing


        try:
            # Get the boolean mask for the current block applied to the current DataFrame
            mask = _get_filter_mask(current_df, block)

            # Apply the boolean mask to the current DataFrame
            current_df = current_df[mask].copy() # Filter the DataFrame and create a new copy for the next iteration
            applied_filters_count += 1
            # Optional: Add a log or print here to show which filter was applied and the resulting shape
            # print(f"Applied filter {i+1} on '{feature_name}' {operation} {value}. New shape: {current_df.shape}")

        except (ValueError, TypeError, RuntimeError) as e:
             st.error(f"Error applying filter block {i+1} ('{feature_name}' {operation} {value}) sequentially: {e}")
             # Decide how to handle errors: skip the block, stop the process, etc.
             # Skipping the block seems reasonable for sequential application.
             continue # Continue to the next filter block
        except Exception as e:
             # Catch any other unexpected errors
             st.error(f"An unexpected error occurred applying filter block {i+1} ('{feature_name}' {operation} {value}) sequentially: {e}")
             continue


    if applied_filters_count == 0 and len(filter_blocks) > 0:
         st.warning("No filter blocks were successfully applied sequentially to the DataFrame.")
         # Return original_df copy if no filters were successfully applied but blocks were defined
         return original_df.copy()
    elif applied_filters_count > 0:
         st.success(f"Successfully applied {applied_filters_count} filter block(s) sequentially.")

    return current_df # Return the final filtered DataFrame

# --- Functions for Merging Section ---
# Keeping the previous merge functions as they are used by the UI to get available tables

def get_merge_available_table_names(raw_datasets: Dict[str, pd.DataFrame], filtered_datasets: Dict[str, pd.DataFrame], merged_tables: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Returns a list of names of all datasets available for merging (raw, filtered, and previously merged).

    Args:
        raw_datasets: A dictionary of raw datasets.
        filtered_datasets: A dictionary of filtered datasets.
        merged_tables: A dictionary of previously merged datasets.

    Returns:
        A list of strings representing the names of all available datasets for merging.
    """
    available_names = list(raw_datasets.keys()) + list(filtered_datasets.keys()) + list(merged_tables.keys())
    return available_names


def perform_merge_operation(left_df: pd.DataFrame, right_df: pd.DataFrame, merge_block: Dict[str, Any]) -> pd.DataFrame:
    """
    Performs a single pandas merge operation based on the provided merge block configuration.

    Args:
        left_df: The left pandas DataFrame.
        right_df: The right pandas DataFrame.
        merge_block: A dictionary containing the merge parameters:
                     - 'how': Type of merge ('inner', 'left', 'right', 'outer', 'cross').
                     - 'on': List with a single column name (optional).
                     - 'left_on': List of column names from the left DataFrame (optional).
                     - 'right_on': List of column names from the right DataFrame (optional).

    Returns:
        The resulting merged pandas DataFrame.

    Raises:
        ValueError: If the merge parameters are invalid or incompatible.
    """
    how = merge_block.get("how", "inner") # Default to inner merge
    on = merge_block.get("on")
    left_on = merge_block.get("left_on")
    right_on = merge_block.get("right_on")

    # Validate merge parameters
    if not on and (not left_on or not right_on):
        if how != 'cross': # Cross merge doesn't require join keys
             raise ValueError("For merge operation, 'on' or both 'left_on' and 'right_on' must be specified (unless 'how' is 'cross').")
    if left_on and right_on and len(left_on) != len(right_on):
         raise ValueError("'left_on' and 'right_on' must have the same length.")
    if on and (left_on or right_on):
         print("Warning: 'on' is specified, 'left_on' and 'right_on' will be ignored.")


    # Prepare merge arguments for pd.merge
    merge_kwargs = {"how": how}

    if on:
        # pd.merge expects 'on' to be a single string or a list of strings
        merge_kwargs["on"] = on if isinstance(on, list) and len(on) > 1 else (on[0] if isinstance(on, list) and len(on) == 1 else on)
        # Ensure 'on' column exists in both dataframes
        if isinstance(merge_kwargs["on"], list):
            for col in merge_kwargs["on"]:
                if col not in left_df.columns or col not in right_df.columns:
                     raise ValueError(f"Column '{col}' specified in 'on' not found in both dataframes.")
        elif isinstance(merge_kwargs["on"], str):
             if merge_kwargs["on"] not in left_df.columns or merge_kwargs["on"] not in right_df.columns:
                  raise ValueError(f"Column '{merge_kwargs['on']}' specified in 'on' not found in both dataframes.")

    elif left_on and right_on:
        merge_kwargs["left_on"] = left_on
        merge_kwargs["right_on"] = right_on
        # Ensure left_on columns exist in left dataframe
        for col in left_on:
            if col not in left_df.columns:
                 raise ValueError(f"Column '{col}' specified in 'left_on' not found in the left dataframe.")
        # Ensure right_on columns exist in right dataframe
        for col in right_on:
            if col not in right_df.columns:
                 raise ValueError(f"Column '{col}' specified in 'right_on' not found in the right dataframe.")

    # Add default suffixes to handle potential column name conflicts
    merge_kwargs["suffixes"] = ('_x', '_y')

    # Perform the merge
    merged_df = pd.merge(left_df, right_df, **merge_kwargs)

    return merged_df

def apply_merge_blocks(datasets: Dict[str, pd.DataFrame], merge_blocks: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Applies a list of merge operations sequentially, using the output of one merge
    as a potential input for the next. Uses all provided datasets as potential starting points.

    Args:
        datasets: A dictionary of all available pandas DataFrames (raw, filtered, previous merges).
                  Keys are dataset names, values are DataFrames.
        merge_blocks: A list of merge block dictionaries, each defining a merge operation.

    Returns:
        A dictionary containing all the resulting merged DataFrames, keyed by their
        'merged_name' as specified in the merge blocks.

    Raises:
        ValueError: If an input table name specified in a merge block is not found
                    in the available datasets.
        Exception: Any error that occurs during the merge operation.
    """
    # Start with all provided datasets as the only available inputs
    available_datasets = datasets.copy()

    merged_results = {} # Dictionary to store the results of each merge block

    for i, merge_block in enumerate(merge_blocks):
        left_table_name = merge_block.get("left_table")
        right_table_name = merge_block.get("right_table")
        merged_name = merge_block.get("merged_name", f"Merged_Result_{i+1}") # Default name

        if not left_table_name or not right_table_name:
             raise ValueError(f"Merge block {i+1}: 'left_table' and 'right_table' names must be specified.")

        # Get the left and right dataframes from the available datasets
        left_df = available_datasets.get(left_table_name)
        right_df = available_datasets.get(right_table_name)
         
        # Check if the input tables are available for merging
        if left_df is None:
            raise ValueError(f"Merge block {i+1}: Left table '{left_table_name}' not found in available datasets for merging.")
        if right_df is None:
            raise ValueError(f"Merge block {i+1}: Right table '{right_table_name}' not found in available datasets for merging.")

        print(f"Applying merge block {i+1}: Merging '{left_table_name}' and '{right_table_name}' as '{merged_name}'")
        try:
            # Perform the single merge operation
            result_df = perform_merge_operation(left_df, right_df, merge_block)

            # Store the result in the merged_results dictionary
            merged_results[merged_name] = result_df

            # Add the result to the available datasets for subsequent merge operations
            available_datasets[merged_name] = result_df

        except Exception as e:
            print(f"Error applying merge block {i+1} ('{merged_name}'): {e}")
            # Depending on how you want to handle errors, you might want to
            # stop processing or continue and log the error.
            # Raising the exception will stop the process.
            raise e

    return merged_results

# --- Reverted functions for Data Transformation Section ---

def get_transformation_operations() -> List[str]:
    """
    Returns a list of supported single feature transformation operations (reverted).

    Returns:
        A list of strings representing the available transformation operations.
    """
    # Reverted to the operations from the initial transformation backend code
    return [
        "Addition",
        "Subtraction",
        "Multiplication",
        "Division",
        "Log", # Reverted to simple Log
        "Square Root",
        "Power",
        "Absolute Value",
        "Rename"
    ]


# --- New Helper Function for Transformations ---
def get_single_feature_transform_operations() -> List[str]:
    """
    Returns a list of supported single feature transformation operations.
    """
    return ["Addition", "Subtraction", "Multiplication", "Division", "Log",
            "Square Root", "Power", "Absolute Value", "Rename", "Rounding"]

# --- Single Feature Transformation Function ---
def apply_single_feature_transform(df: pd.DataFrame, transform_block: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a single feature transformation to a DataFrame and adds the result as a new column.

    Args:
        df: The input pandas DataFrame.
        transform_block: A dictionary defining the transformation, containing:
                         - 'feature': The name of the column to transform.
                         - 'operation': The type of transformation operation (str).
                         - 'value': The value to use for the operation (e.g., for addition, power).
                                    This will be None for operations like Log, Abs, Sqrt, Rename, Rounding.
                         - 'output_name': The name for the new transformed column.

    Returns:
        A new DataFrame with the transformed column added.

    Raises:
        ValueError: If the input feature is not found, the operation is invalid for the feature type,
                    or an output name conflict occurs that cannot be resolved automatically (if we chose to raise).
        TypeError: If the feature's data type is incompatible with the chosen operation.
        Exception: Any other unexpected error during transformation.
    """
    feature = transform_block.get("feature")
    operation = transform_block.get("operation")
    value = transform_block.get("value")
    output_name = transform_block.get("output_name")

    # Basic validation for essential block components
    if not feature or not operation or not output_name:
        raise ValueError(f"Transformation block incomplete: missing feature, operation, or output name. Block: {transform_block}")

    # Check if the specified feature exists in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame passed into this function call
    transformed_df = df.copy()

    # Handle output column name conflicts. If output_name exists and is not the original feature
    # (this prevents self-renaming from causing a conflict), append a suffix.
    if output_name in transformed_df.columns and output_name != feature:
        original_output_name = output_name
        k = 1
        while f"{original_output_name}_{k}" in transformed_df.columns:
            k += 1
        output_name = f"{original_output_name}_{k}"
        # Note: We don't raise an error here because the backend automatically resolves it.
        # The UI could add a warning if this automatic renaming happens.

    # Get the Series for the feature to be transformed
    feature_series = transformed_df[feature]

    # Helper function for numeric type checking
    def _check_numeric(series_to_check, op_name):
        if not pd.api.types.is_numeric_dtype(series_to_check):
            raise TypeError(f"Operation '{op_name}' requires a numeric feature. Feature '{feature}' is of type '{series_to_check.dtype}'.")

    try:
        if operation == "Addition":
            _check_numeric(feature_series, "Addition")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for 'Addition' must be a number. Received: {type(value)}")
            transformed_df[output_name] = feature_series + value
        elif operation == "Subtraction":
            _check_numeric(feature_series, "Subtraction")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for 'Subtraction' must be a number. Received: {type(value)}")
            transformed_df[output_name] = feature_series - value
        elif operation == "Multiplication":
            _check_numeric(feature_series, "Multiplication")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for 'Multiplication' must be a number. Received: {type(value)}")
            transformed_df[output_name] = feature_series * value
        elif operation == "Division":
            _check_numeric(feature_series, "Division")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for 'Division' must be a number. Received: {type(value)}")
            if value == 0:
                # Handle division by zero: results in NaN where numerator is 0, else Inf
                transformed_df[output_name] = np.where(feature_series == 0, np.nan, feature_series / value)
                raise ValueError(f"Division by zero attempted for feature '{feature}'. Resulting values might be NaN or Inf.")
            else:
                transformed_df[output_name] = feature_series / value
        elif operation == "Log":
            _check_numeric(feature_series, "Log")
            # np.log handles non-positive values (log(0) is -inf, log(negative) is NaN).
            # We raise a ValueError if such values are present to inform the user.
            transformed_df[output_name] = np.log(feature_series)
            if (feature_series <= 0).any():
                raise ValueError(f"Log operation encountered non-positive values for feature '{feature}'. Resulting values might be NaN or -Inf.")
        elif operation == "Square Root":
            _check_numeric(feature_series, "Square Root")
            # np.sqrt handles negative values (results in NaN). Clip at 0 to explicitly avoid NaN for negative inputs.
            transformed_df[output_name] = np.sqrt(feature_series.clip(lower=0))
            if (feature_series < 0).any():
                raise ValueError(f"Square Root operation encountered negative values for feature '{feature}'. Values were clipped to 0, resulting in 0 or NaN.")
        elif operation == "Power":
            _check_numeric(feature_series, "Power")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for 'Power' must be a number. Received: {type(value)}")
            transformed_df[output_name] = np.power(feature_series, value)
        elif operation == "Absolute Value":
            _check_numeric(feature_series, "Absolute Value")
            transformed_df[output_name] = np.abs(feature_series)
        elif operation == "Rename":
            # For rename, we copy the original column to the new output_name.
            # The original column under its old name is kept unless explicitly dropped later.
            transformed_df[output_name] = feature_series.copy()
        elif operation == "Rounding":
            _check_numeric(feature_series, "Rounding")
            # Round to 2 decimal places as specified
            transformed_df[output_name] = np.round(feature_series, 2)
        else:
            raise ValueError(f"Unsupported transformation operation: '{operation}'.")

    except (ValueError, TypeError) as e:
        # Re-raise specific ValueErrors and TypeErrors for frontend to catch
        raise e
    except Exception as e:
        # Catch any other unexpected exceptions and re-raise as a generic error
        raise Exception(f"An unexpected error occurred during transformation '{operation}' on feature '{feature}': {e}")

    return transformed_df


def apply_multi_feature_transform(df: pd.DataFrame, transform_block: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a multi-feature transformation to a DataFrame and adds the result as a new column.

    Args:
        df: The input pandas DataFrame.
        transform_block: A dictionary defining the transformation, containing:
                         - 'features': A list of names of the columns to combine.
                         - 'operation': The type of combination operation (str, e.g., 'Sum', 'Mean').
                         - 'output_name': The name for the new combined column.

    Returns:
        A new DataFrame with the combined column added.
        Returns the original DataFrame if the transformation cannot be applied.

    Raises:
        ValueError: If input features are not found or the operation is invalid for the feature types.
        Exception: Any error that occurs during the transformation.
    """
    features = transform_block.get("features")
    operation = transform_block.get("operation")
    output_name = transform_block.get("output_name")

    if not features or not operation or not output_name:
        print(f"Skipping invalid multi-transformation block: {transform_block}")
        return df.copy() # Return a copy

    # Check if all input features exist
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Input feature '{feature}' not found in the DataFrame for multi-feature transformation.")

    # Create a copy to avoid modifying the original DataFrame
    transformed_df = df.copy()

    # Check if output column name already exists
    if output_name in transformed_df.columns:
         # Option 3: Append a suffix (safer)
         original_output_name = output_name
         k = 1
         while output_name in transformed_df.columns:
              output_name = f"{original_output_name}_{k}"
              k += 1
         print(f"Warning: Output column name '{original_output_name}' already exists. Using '{output_name}' instead.")


    try:
        # Select the columns to operate on
        data_subset = transformed_df[features]

        # Perform the combination based on the operation
        if operation.lower() == "sum":
            # Ensure all selected features are numeric for sum
            if not all(pd.api.types.is_numeric_dtype(data_subset[col]) for col in features):
                 raise ValueError("Sum operation requires all selected features to be numeric.")
            transformed_df[output_name] = data_subset.sum(axis=1)
        elif operation.lower() == "mean":
            # Ensure all selected features are numeric for mean
            if not all(pd.api.types.is_numeric_dtype(data_subset[col]) for col in features):
                 raise ValueError("Mean operation requires all selected features to be numeric.")
            transformed_df[output_name] = data_subset.mean(axis=1)
        elif operation.lower() == "product":
            # Ensure all selected features are numeric for product
            if not all(pd.api.types.is_numeric_dtype(data_subset[col]) for col in features):
                 raise ValueError("Product operation requires all selected features to be numeric.")
            transformed_df[output_name] = data_subset.prod(axis=1)
        elif operation.lower() == "max":
            # Ensure all selected features are numeric for max
            if not all(pd.api.types.is_numeric_dtype(data_subset[col]) for col in features):
                 raise ValueError("Max operation requires all selected features to be numeric.")
            transformed_df[output_name] = data_subset.max(axis=1)
        elif operation.lower() == "min":
            # Ensure all selected features are numeric for min
            if not all(pd.api.types.is_numeric_dtype(data_subset[col]) for col in features):
                 raise ValueError("Min operation requires all selected features to be numeric.")
            transformed_df[output_name] = data_subset.min(axis=1)
        # Add more multi-feature operations here as needed
        else:
            raise ValueError(f"Unsupported multi-feature transformation operation: '{operation}'.")

    except (ValueError, TypeError) as e:
        print(f"Multi-feature transformation configuration error for features '{features}' with operation '{operation}': {e}")
        # Re-raise the specific error
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during multi-feature transformation for features '{features}' with operation '{operation}': {e}")
        # Re-raise the unexpected error
        raise e


# --- NEW: Functions for AI-driven Multi-Feature Transformation Section ---

def get_multi_feature_ai_operations() -> List[str]:
    """
    Returns a conceptual list of operations for AI-driven multi-feature transformations.
    In this setup, the user provides free-form text, which an AI model would interpret.
    This list is primarily for UI guidance, indicating that complex operations are possible.
    """
    return [
        "User-defined (e.g., 'sum of X and Y', 'average of A, B, C', 'product of all features')",
        # More examples can be added here for UI guidance
    ]


def generate_transform_code_with_llm(features: List[str], user_operation_text: str) -> str:
    """
    Delegates code generation to the GenAIAgent.
    """
    return gen_ai_agent.generate_transform_code(features, user_operation_text)


def apply_ai_driven_multi_feature_transform(df: pd.DataFrame, transform_block: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a single AI-driven multi-feature transformation to a DataFrame
    and adds the result as a new column.

    Args:
        df: The input pandas DataFrame (e.g., output from single-feature transformations).
        transform_block: A dictionary defining the transformation, containing:
                         - 'features': A list of names of the columns to combine.
                         - 'operation': The free-form text string entered by the user (AI input).
                         - 'output_name': The name for the new transformed column.

    Returns:
        A new DataFrame with the transformed column added.

    Raises:
        ValueError: If input features are not found, the operation text is invalid,
                    or an output name conflict occurs that cannot be resolved automatically.
        TypeError: If data types are incompatible with the AI-interpreted operation.
        Exception: Any other unexpected error during transformation.
    """
    features = transform_block.get("features")
    user_operation_text = transform_block.get("operation") # This is now the free-form text
    output_name = transform_block.get("output_name")

    # Basic validation for essential block components
    if not features or not user_operation_text or not output_name:
        raise ValueError(f"AI-driven transformation block incomplete: missing features, operation text, or output name. Block: {transform_block}")

    # Check if all input features exist
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Input feature '{feature}' not found in the DataFrame for AI-driven multi-feature transformation.")

    # Create a copy to avoid modifying the original DataFrame passed into this function call
    transformed_df = df.copy()

    # Handle output column name conflicts by appending a suffix
    if output_name in transformed_df.columns:
        original_output_name = output_name
        k = 1
        while f"{original_output_name}_{k}" in transformed_df.columns:
            k += 1
        output_name = f"{original_output_name}_{k}"
        st.warning(f"Output column name '{original_output_name}' already exists. Using '{output_name}' instead.")

    try:
        # >>>>> MODIFIED LINE: Call the new code generation function <<<<<
        generated_code = generate_transform_code_with_llm(features, user_operation_text)

        if generated_code.startswith("ERROR:"):
            raise ValueError(f"AI interpretation failed for '{user_operation_text}': {generated_code}")

        # >>>>> ADDED: Execute the generated code safely <<<<<
        execution_context = {
            'df': transformed_df, # Pass the transformed_df (copy) into the context
            'np': np,
            'pd': pd
        }
        # The LLM is instructed to generate a Series.
        result_series = eval(generated_code, {}, execution_context)

        if not isinstance(result_series, pd.Series):
            raise TypeError(f"AI generated code did not return a pandas Series. Got: {type(result_series)}. Generated code: '{generated_code}'")

        if np.isinf(result_series).any():
             result_series = result_series.replace([np.inf, -np.inf], np.nan)


        # Add the new column to the DataFrame
        transformed_df[output_name] = result_series

    except SyntaxError as se:
        st.error(f"Error: AI generated invalid Python code: {generated_code}. Details: {se}")
        raise ValueError(f"AI generated invalid Python code: {generated_code}. Error: {se}")
    except NameError as ne:
        st.error(f"Error: AI generated code referenced an unknown name (e.g., a feature that doesn't exist or is misspelled by AI): {generated_code}. Details: {ne}")
        raise ValueError(f"AI generated code referenced an unknown name (e.g., a feature that doesn't exist or is misspelled by AI): {generated_code}. Error: {ne}")
    except TypeError as te:
        st.error(f"Error: AI generated code resulted in a type error (e.g., trying to sum text columns): {generated_code}. Details: {te}")
        raise ValueError(f"AI generated code resulted in a type error (e.g., trying to sum text columns): {generated_code}. Error: {te}")
    except Exception as e:
        # Catch any other unexpected exceptions and re-raise as a generic error
        st.error(f"An unexpected error occurred during AI-driven multi-feature transformation for '{user_operation_text}'. Generated code: '{generated_code}'. Details: {e}")
        raise Exception(f"An unexpected error occurred during AI-driven multi-feature transformation for '{user_operation_text}': {e}")

    return transformed_df


def apply_all_ai_driven_multi_feature_transforms(original_df: pd.DataFrame, transform_blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Applies a list of AI-driven multi-feature transformation blocks sequentially to a DataFrame.
    Each subsequent transformation is applied to the result of the previous transformation.

    Args:
        original_df: The initial DataFrame to apply transformations to (e.g., output from single-feature transforms).
        transform_blocks: A list of dictionaries, where each dictionary defines an AI-driven
                          multi-feature transformation block.

    Returns:
        A new DataFrame containing the original data plus all newly created transformed columns.
        Returns the original DataFrame if transform_blocks is empty.
    """
    if not transform_blocks:
        return original_df.copy() # Return a copy if no transformations

    current_df = original_df.copy()
    applied_transforms_count = 0

    for i, block in enumerate(transform_blocks):
        features = block.get("features")
        user_operation_text = block.get("operation")
        output_name = block.get("output_name")

        # Skip incomplete blocks
        if not features or not user_operation_text or not output_name:
            st.warning(f"Skipping incomplete AI-driven multi-feature transform block {i+1}: {block}")
            continue

        # Check for feature existence in current_df columns BEFORE applying the transform
        missing_features = [f for f in features if f not in current_df.columns]
        if missing_features:
            st.warning(f"Skipping AI-driven multi-feature transform block {i+1} - "
                       f"Features {missing_features} not found in current intermediate data.")
            continue

        print(f"Applying AI-driven multi-feature transform block {i+1}: "
              f"Features: {features}, Operation: '{user_operation_text}', Output: '{output_name}'")
        try:
            current_df = apply_ai_driven_multi_feature_transform(current_df, block)
            applied_transforms_count += 1
            st.success(f"Successfully applied AI-driven multi-feature transform block {i+1}.")
        except (ValueError, TypeError, Exception) as e:
            # Errors are already logged by apply_ai_driven_multi_feature_transform
            st.error(f"Failed to apply AI-driven multi-feature transform block {i+1}. Error: {e}")
            continue # Continue to the next block even if one fails

    if applied_transforms_count == 0 and len(transform_blocks) > 0:
        st.warning("No AI-driven multi-feature transform blocks were successfully applied.")
        return original_df.copy() # Return original_df copy if no transforms applied but blocks were defined
    elif applied_transforms_count > 0:
        st.info(f"Completed applying {applied_transforms_count} AI-driven multi-feature transform block(s).")

    return current_df


# --- NEW: Mandatory Feature Selection Function ---
def select_mandatory_features(df: pd.DataFrame) -> List[str]:

    """
    Simulates a model assessing and selecting mandatory features from the given DataFrame.
    For now, it returns a fixed list of features if they are present in the DataFrame.
    In a real scenario, this would involve a more complex feature importance model.

    Args:
        df (pd.DataFrame): The input DataFrame (after target variable removal).

    Returns:
        List[str]: A list of selected mandatory feature names.
    """
    if df.empty:
        return []

    # Define a set of "pre-determined" mandatory features for simulation
    potential_mandatory_features = [
        "OPB", "ApplicationID" , "CREDIT_SCORE_AVG_CALC","DELINQ_CNT_30_DAY_TOTAL","Timestamp_x","TERM_OF_LOAN","PREPAYMENT_EVENT_LABEL","COF_EVENT_LABEL"
    ]

    # Filter this list to include only features actually present in the input DataFrame
    present_mandatory_features = [
        feat for feat in potential_mandatory_features if feat in df.columns
    ]

    # For simulation, let's randomly pick 5 to 7 features from the present ones
    # to simulate a selection process, ensuring we don't pick more than available.
    num_to_select = min(random.randint(5, 7), len(present_mandatory_features))
    selected_mandatory = random.sample(present_mandatory_features, num_to_select)

    return selected_mandatory

# --- NEW: Recommend Features Function ---
def recommend_features(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Simulates a Gen AI model recommending features from the input DataFrame.
    It selects a random subset of features and provides simulated descriptions and stats.

    Args:
        df (pd.DataFrame): The input DataFrame (e.g., combined_dataset after merging).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict contains
                              'Feature', 'Description', 'Min', 'Max', 'Mean', 'Data Type'.
    """
    if df.empty:
        return []

    available_features = df.columns.tolist()
    if not available_features:
        return []

    # Decide how many features to recommend (e.g., 5 to 10)
    num_to_recommend = min(random.randint(5, 10), len(available_features))
    recommended_feature_names = random.sample(available_features, num_to_recommend)

    recommended_features_info = []
    # General placeholder descriptions for various feature types
    default_descriptions = {
        "numeric": "A continuous numerical attribute.",
        "categorical": "A categorical attribute representing different groups.",
        "boolean": "A binary attribute (True/False or 0/1).",
        "datetime": "A date or time attribute."
    }

    for feature in recommended_feature_names:
        data = df[feature]
        description = f"AI-recommended feature: {feature_descriptions.get(feature, default_descriptions.get('numeric', ''))}"
        
        # Simulate generic descriptions if not already defined
        if "OPB" in feature: description = "Outstanding Principal Balance of the customer's loan (AI-recommended)"
        elif "interest_rate" in feature: description = "Current interest rate applicable to the loan (AI-recommended)"
        elif "tenure" in feature: description = "Duration of the loan in months (AI-recommended)"
        elif "credit_score_band" in feature: description = "Customer's credit score category (AI-recommended)"
        elif "LTV" in feature: description = "Loan-to-Value ratio (AI-recommended)"
        elif "age" in feature: description = "Customer's age in years (AI-recommended)"
        elif "income" in feature: description = "Customer's annual income (AI-recommended)"
        elif "loan_amount" in feature: description = "Original loan amount (AI-recommended)"
        elif "loan_type" in feature: description = "Type of loan (Personal, Mortgage, etc.) (AI-recommended)"
        else:
            description = f"AI-recommended feature derived from original data: {feature}"


        stats = {
            'Feature': feature,
            'Description': description
        }

        if pd.api.types.is_numeric_dtype(data):
            stats['Min'] = f"{data.min():.2f}"
            stats['Max'] = f"{data.max():.2f}"
            stats['Mean'] = f"{data.mean():.2f}"
            stats['Data Type'] = 'Numeric'
        else:
            stats['Min'] = 'N/A'
            stats['Max'] = 'N/A'
            stats['Mean'] = 'N/A'
            stats['Data Type'] = 'Categorical' # Or 'Object', 'Boolean', 'DateTime'

        recommended_features_info.append(stats)

    return recommended_features_info

# Helper descriptions for existing features (can be expanded)
feature_descriptions = {
    "OPB": "Outstanding Principal Balance of the customer's loan",
    "interest_rate": "Current interest rate applicable to the customer's loan",
    "tenure": "Duration of the loan in months",
    "credit_score_band": "Customer's credit score category (Excellent, Good, Fair, Poor)",
    "LTV": "Loan-to-Value ratio indicating the risk level of the loan",
    "age": "Customer's age in years",
    "income": "Customer's annual income",
    "employment_length": "Length of employment in years",
    "debt_to_income": "Ratio of total debt to income",
    "payment_history": "Customer's payment history score",
    "loan_amount": "Original loan amount",
    "loan_type": "Type of loan (Personal, Mortgage, Auto, Business)",
    "property_value": "Value of the property (for mortgage loans)",
    "down_payment": "Amount of down payment made",
    "loan_purpose": "Purpose of the loan",
    "marital_status": "Customer's marital status",
    "education": "Customer's education level",
    "residence_type": "Type of residence (Own, Rent, Other)",
    "number_of_dependents": "Number of dependents",
    "previous_loans": "Number of previous loans"
}