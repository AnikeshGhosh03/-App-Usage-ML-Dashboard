# -App-Usage-ML-Dashboard
This project is a Streamlit-based interactive dashboard that leverages machine learning to analyze app usage patterns and identify how different apps may impact your sleep quality or productivity.

🚀 Features:
Upload Excel files containing app usage logs,
Automatically preprocesses and scales data,
Trains a Random Forest classifier with hyperparameter tuning,
Displays overall model accuracy (e.g., 87%)

📊 Shows:
Confusion Matrix for predictions,
Classification Report,
Feature Importance of app usage metrics

🔥 App-wise Sleep Harm Risk Visualization:
Flags “high-risk” apps predicted as non-productive,
Highlights “safe” apps predicted as productive,
Sends warning alerts for apps used excessively

📥 Downloadable CSV of predictions

💡 Clean, interactive, user-friendly Streamlit UI

🧠 ML Details:
Model: Random Forest Classifier,
Hyperparameter tuning using GridSearchCV,
Class imbalance handled using SMOTE,
Accuracy achieved: ~87% (based on sample dataset)

📊 Use Case:
Designed to help users or researchers understand the effect of app usage behavior (social, productivity, entertainment, etc.) on sleep patterns or productivity through data-driven insights.