from src.ingestion import load_and_clean_data
from src.engineering import engineer_features
from src.training import train_model
from src.reporting import generate_report

def run_pipeline():
    print("🚀 Starting Sentinel Fraud Detection Pipeline")
    
    # 1. Data Ingestion
    load_and_clean_data('data/data.csv')
    
    # 2. Feature Engineering (Fixing Data Leakage)
    engineer_features('data/cleaned_data.csv')
    
    # 3. Model Training
    train_model('data/features_data.csv')
    
    # 4. Visual Reporting
    generate_report('models/sentinel_model.pkl', 'data/features_data.csv')
    
    print("✅ Pipeline execution finished successfully")

if __name__ == "__main__":
    run_pipeline()