import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(input_path):
    print(f"--- Training Model with {input_path} ---")
    df = pd.read_csv(input_path)
    
    X = df.drop('Fraud_Label', axis=1)
    y = df['Fraud_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\n--- Performance Report ---")
    print(classification_report(y_test, predictions))
    
    joblib.dump(model, 'models/sentinel_model.pkl')
    print("--- Model saved to models/sentinel_model.pkl ---")

if __name__ == "__main__":
    train_model('data/features_data.csv')