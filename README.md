# Sentinel: End-to-End Fraud Detection Pipeline 🛡️

**Sentinel** is a modular Machine Learning pipeline built to identify fraudulent financial transactions. Unlike traditional monolithic notebooks, this project implements a **production-ready architecture** focused on scalability, data integrity, and clear separation of concerns.

## 🚀 Project Architecture
The system is organized into a modular pipeline to ensure maintainability:
- **`src/ingestion.py`**: Handles raw data loading, timestamp standardization, and initial cleaning.
- **`src/engineering.py`**: Executes feature transformation, categorical encoding, and creates advanced risk indicators.
- **`src/training.py`**: Trains a Random Forest Classifier and generates performance reports.
- **`main.py`**: The central orchestrator that executes the full pipeline with a single command.

## 🧠 The "Data Leakage" Case Study
During the initial development phase, the model achieved a suspicious 100% precision. Through rigorous data profiling, I identified **`Risk_Score`** as a "Target Leak"—a feature that contained information about the outcome which wouldn't be available in a real-time production environment.

**The Fix:** I refactored the pipeline to drop the `Risk_Score`, forcing the model to learn from raw behavioral patterns:
- **Transaction Velocity**: Frequency of actions within time windows.
- **Spending Deviations**: Ratio of current transaction amount vs. 7-day average.
- **Geographic Anomalies**: Distance-based risk assessment.

This resulted in a realistic, robust model ready for real-world deployment.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Data & ML**: Pandas, Scikit-Learn, Joblib
- **Utilities**: Python-Dateutil, PyTZ
- **Architecture**: Modular Scripting

## 📦 Installation & Usage
1. Clone this repository:
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/Fraud-detection.git
2. Install the required environment:
    ```bash
    pip install -r requirements.txt
3. Execute the full pipeline:
    ```bash
    python main.py
📊 Evaluation Metrics
The pipeline prioritizes Recall for the fraud class to minimize financial loss, while maintaining high Precision to avoid friction for legitimate users.

Developed By Ulises Chustek