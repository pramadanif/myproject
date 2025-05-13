import pandas as pd
import os

def remove_columns(input_file, output_file, columns_to_remove):
    """
    Remove specified columns from a CSV file and save to a new file
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        columns_to_remove (list): List of column names to remove
    """
    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Print original columns
    print(f"Original columns: {df.columns.tolist()}")
    
    # Check if the columns to remove exist
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    if existing_columns:
        # Remove the specified columns
        print(f"Removing columns: {existing_columns}")
        df = df.drop(columns=existing_columns)
        
        # Save the modified DataFrame to a new CSV file
        print(f"Saving modified file to: {output_file}")
        df.to_csv(output_file, index=False)
        print("Done! Columns have been removed successfully.")
    else:
        print(f"Warning: None of the specified columns {columns_to_remove} were found in the CSV file.")
        print(f"Available columns are: {df.columns.tolist()}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_file = os.path.join(os.path.dirname(__file__), 'indoperkasa2.csv')
    output_file = os.path.join(os.path.dirname(__file__), 'indoperkasa2_modified.csv')
    
    # Define the columns to remove
    columns_to_remove = ['Total_Usage_Score', 'Kondisi_Oli_Encoded', 'Kondisi_Rem_Encoded']
    
    # Remove the columns
    remove_columns(input_file, output_file, columns_to_remove)
