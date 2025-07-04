import pandas as pd
import numpy as np
import os

# --- Configuration ---
FILE_PATH = 'notebook\data\personality_dataset_copy.csv'
FILE_NAME = 'personality_dataset.csv'
NEW_FILENAME = 'augmented_dataset.csv'
TARGET_COLUMN = 'Personality'
MINORITY_CLASS = 'Introvert'
MAJORITY_CLASS = 'Extrovert'

def augment_data():
    """
    Reads the dataset, generates synthetic data for the minority class,
    and appends it back to the original CSV file.
    """
    if not os.path.exists(FILE_PATH):
        print(f"Error: The file '{FILE_NAME}' was not found in the current directory.")
        return

    # 1. Load the dataset
    print(f"Reading data from '{FILE_NAME}'...")
    df = pd.read_csv(FILE_PATH)

   # 2. Clean the data by removing duplicates (as you specified)
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    cleaned_rows = len(df)
    duplicates_removed = initial_rows - cleaned_rows
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # --- Analysis Step (on cleaned data) ---
    print("\n--- Cleaned Dataset Analysis ---")
    class_counts = df[TARGET_COLUMN].value_counts()
    print("Current class distribution:")
    print(class_counts)

    minority_count = class_counts.get(MINORITY_CLASS, 0)
    majority_count = class_counts.get(MAJORITY_CLASS, 0)

    if minority_count == 0:
        print(f"Error: Minority class '{MINORITY_CLASS}' not found in the dataset.")
        return
    if minority_count >= majority_count:
        print("\nDataset is already balanced or the minority class is larger. No augmentation needed.")
        return

    # Determine how many samples to generate
    samples_to_generate = majority_count - minority_count
    print(f"\nGoal: Generating {samples_to_generate} new samples for the '{MINORITY_CLASS}' class to balance the dataset.")

    # 3. Isolate the minority class data to learn its characteristics
    minority_df = df[df[TARGET_COLUMN] == MINORITY_CLASS]

    # --- Pre-calculate distributions for generating realistic data ---
    # For categorical columns, calculate the probability of each category
    categorical_cols = minority_df.select_dtypes(include=['object', 'category']).columns.drop(TARGET_COLUMN)
    prob_distributions = {}
    for col in categorical_cols:
        counts = minority_df[col].value_counts(normalize=True)
        prob_distributions[col] = {'values': counts.index, 'probs': counts.values}

    # For numerical columns, get min and max to create a realistic range
    numerical_cols = minority_df.select_dtypes(include=np.number).columns
    numerical_ranges = {col: (minority_df[col].min(), minority_df[col].max()) for col in numerical_cols}

    # 4. Generate synthetic data
    print("Generating synthetic data...")
    new_rows = []
    for _ in range(samples_to_generate):
        new_row = {}
        # Generate numerical features based on the observed min/max range
        for col, (min_val, max_val) in numerical_ranges.items():
            new_row[col] = np.random.uniform(min_val, max_val)

        # Generate categorical features based on observed probabilities
        for col, dist in prob_distributions.items():
            new_row[col] = np.random.choice(dist['values'], p=dist['probs'])

        # Set the target class for the new row
        new_row[TARGET_COLUMN] = MINORITY_CLASS
        new_rows.append(new_row)

    # 5. Combine the cleaned data with the new synthetic data
    if not new_rows:
        print("No new rows were generated.")
        return
        
    synthetic_df = pd.DataFrame(new_rows)
    # Use the cleaned 'df' as the base
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)

    # 5. Save the augmented data back to the original file
    try:
        combined_df.to_csv(NEW_FILENAME, index=False)
        print(f"\nSuccessfully generated {len(synthetic_df)} new samples.")
        print(f"The file '{FILE_NAME}' has been updated.")

        # --- Verification Step ---
        print("\n--- New Dataset Analysis ---")
        print("New class distribution:")
        print(combined_df[TARGET_COLUMN].value_counts())
        print("\nProcess complete.")

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
        print("Your original file has not been changed.")


if __name__ == '__main__':
    # Add a safety check prompt
    print("WARNING: This script will modify your CSV file in place.")
    print(f"Please make a backup of '{FILE_NAME}' before proceeding.")
    
    user_input = input("Do you want to continue? (yes/no): ").lower()
    
    if user_input == 'yes':
        augment_data()
    else:
        print("Operation cancelled by the user.")