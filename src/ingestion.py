import pandas as pd
import os

def load_and_clean_data(file_path):
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_csv(file_path)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop_duplicates(inplace=True)
    
    output_path = 'data/cleaned_data.csv'
    df.to_csv(output_path, index=False)
    print(f"--- Data cleaned and saved to {output_path} ---")

if __name__ == "__main__":
    load_and_clean_data('data/data.csv')