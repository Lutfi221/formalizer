# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SacreBLEU Score Calculation from CSV (using `evaluate` package)
#
# This notebook calculates the SacreBLEU score using predictions and references from a CSV file.
# It utilizes the `evaluate` library from Hugging Face.
# The CSV file is expected to have at least two columns:
# 1.  `formal_reference`: Containing the ground truth formal sentences.
# 2.  `predicted_formal`: Containing the model's predicted formal sentences.

# %%
# %pip install pandas evaluate sacrebleu --quiet
# sacrebleu is often a dependency for evaluate's sacrebleu module,
# but installing it explicitly ensures it's available.
# You can also try !pip install pandas evaluate --quiet

# %%
import pandas as pd
import evaluate # Use the evaluate library
from pathlib import Path

# %% [markdown]
# ## 1. Configuration
#
# Please specify the path to your CSV file and the names of the columns.

# %%
# --- Configuration ---
# Set the path to your CSV file
# Example: csv_file_path_str = "path/to/your/results.csv"
# Or use Path object for better path handling:
# csv_file_path = Path("path/to/your/results.csv")

# csv_file_path_str = input("Enter the path to your CSV file: ")
csv_file_path_str = "/Users/lutfi/neodata/projects/formalizer/data/validation_predictions.csv"
csv_file_path = Path(csv_file_path_str)

# Define the column names
reference_column = "formal_reference"
prediction_column = "predicted_formal"
original_column = "informal" # Defined for clarity, but not used in BLEU calculation


# %% [markdown]
# ## 2. Load Data and Calculate BLEU Score

# %%
def calculate_bleu_from_csv_evaluate(file_path: Path, ref_col: str, pred_col: str):
    """
    Loads data from a CSV file and calculates the SacreBLEU score using the evaluate package.

    Args:
        file_path (Path): Path to the CSV file.
        ref_col (str): Name of the column containing reference sentences.
        pred_col (str): Name of the column containing predicted sentences.

    Returns:
        dict or None: A dictionary containing the BLEU score and other metrics,
                      or None if an error occurs.
    """
    try:
        print(f"Attempting to load data from: {file_path}")
        if not file_path.is_file():
            print(f"Error: File not found at {file_path}")
            return None

        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")

        # Validate required columns
        if ref_col not in df.columns:
            print(f"Error: Reference column '{ref_col}' not found in the CSV.")
            return None
        if pred_col not in df.columns:
            print(f"Error: Prediction column '{pred_col}' not found in the CSV.")
            return None

        # Extract references and predictions
        # Ensure they are lists of strings. Handle potential NaN values by converting to empty strings.
        references = [[str(r) if pd.notna(r) else ""] for r in df[ref_col].tolist()]
        predictions = [str(p) if pd.notna(p) else "" for p in df[pred_col].tolist()]

        if not references or not predictions:
            print("Error: No references or predictions found after processing.")
            return None
        
        if len(references) != len(predictions):
            print(f"Warning: Mismatch in number of references ({len(references)}) and predictions ({len(predictions)}).")
            # BLEU calculation will still proceed with the available pairs.

        print(f"\nLoading SacreBLEU metric from 'evaluate' package...")
        sacrebleu_metric = evaluate.load("sacrebleu")
        print("Metric loaded successfully.")

        print(f"Calculating BLEU score for {len(predictions)} sentences...")

        # Calculate BLEU score using evaluate
        # compute() expects predictions as a list of strings and references as a list of lists of strings
        results = sacrebleu_metric.compute(predictions=predictions, references=references, lowercase=True)
        
        return results

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return None

# %% [markdown]
# ## 3. Execute Calculation and Display Results

# %%
if __name__ == "__main__":
    if 'csv_file_path' in locals() and csv_file_path.name and csv_file_path_str.strip(): # Check if path is not empty string
        bleu_results_dict = calculate_bleu_from_csv_evaluate(csv_file_path, reference_column, prediction_column)
        # bleu_results_dict = calculate_bleu_from_csv_evaluate(csv_file_path, reference_column, original_column)

        if bleu_results_dict:
            print("\n--- SacreBLEU Score (via `evaluate` package) ---")
            print(f"BLEU Score: {bleu_results_dict.get('score', 0.0):.2f}") # Score is 0-100
            print("\n--- Detailed Metrics ---")
            for key, value in bleu_results_dict.items():
                if key == 'precisions':
                    print(f"  Precisions: {[f'{p:.4f}' for p in value]}")
                elif isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("CSV file path was not provided or is invalid. Please set 'csv_file_path_str' by running the configuration cell in section 1.")

# %% [markdown]
# ### Understanding `evaluate.load("sacrebleu")` Output:
#
# The `.compute()` method returns a dictionary. Key metrics include:
#
# * **`score`**: The final corpus-level BLEU score (0-100).
# * **`counts`**: List of matching n-gram counts for each order (1-gram, 2-gram, etc.).
# * **`totals`**: List of total n-grams in predictions for each order.
# * **`precisions`**: List of precision for each n-gram order.
# * **`bp` (or `brevity_penalty`)**: Brevity penalty. A BP of 1.0 means no penalty.
# * **`sys_len`**: Total length (tokens) of predicted sentences.
# * **`ref_len`**: Total length (tokens) of reference sentences.
# * **`tokenize`**: The tokenizer used by SacreBLEU (e.g., '13a', 'zh', 'intl').
# * **`smooth_method`**: The smoothing method applied (if any).
