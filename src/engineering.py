import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(input_path):
    print(f"--- Feature Engineering on {input_path} ---")
    df = pd.read_csv(input_path)
    
    # Financial signals
    df['Amount_to_Avg_Ratio'] = df['Transaction_Amount'] / (df['Avg_Transaction_Amount_7d'] + 0.1)
    
    # Category Encoding
    le = LabelEncoder()
    categorical_cols = ['Transaction_Type', 'Device_Type', 'Location', 
                        'Merchant_Category', 'Card_Type', 'Authentication_Method']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # DROPPING RISK_SCORE TO AVOID LEAKAGE
    cols_to_drop = ['Transaction_ID', 'User_ID', 'Timestamp', 'Risk_Score']
    df = df.drop(columns=cols_to_drop)
    
    output_path = 'data/features_data.csv'
    df.to_csv(output_path, index=False)
    print(f"--- Engineering complete. Saved to {output_path} ---")

if __name__ == "__main__":
    engineer_features('data/cleaned_data.csv')