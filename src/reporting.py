import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_report(model_path, data_path):
    print("--- Generating Visual Analytics ---")
    
    # Ensure the reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df.drop('Fraud_Label', axis=1)
    y = df['Fraud_Label']
    
    # 1. Feature Importance Plot
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    # Fixed the palette warning by assigning hue
    sns.barplot(x='importance', y='feature', data=importances, hue='feature', palette='magma', legend=False)
    plt.title('Sentinel: Key Fraud Drivers (Feature Importance)')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    
    # 2. Confusion Matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud']).plot(cmap='Blues', ax=ax)
    plt.title('Sentinel: Prediction Accuracy Matrix')
    plt.savefig('reports/confusion_matrix.png')
    
    print("✅ Visual reports saved to the /reports directory")

if __name__ == "__main__":
    generate_report('models/sentinel_model.pkl', 'data/features_data.csv')