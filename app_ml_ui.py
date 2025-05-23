import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import io

st.set_page_config(page_title="📱 App Usage ML Analyzer", layout="wide")
st.title("📊 ML Classifier for App Screen Time Analysis")

uploaded_file = st.file_uploader("📁 Upload your Excel Dataset", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Preview full dataset
        st.write("### 📄 Full Dataset", df)

        # Drop date or categorical if not encoded
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)

        # Use fixed target column
        target_column = "is_productive"
        if target_column not in df.columns:
            st.error("❌ 'is_productive' column not found in dataset.")
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Encode target if needed
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            # Standardize features
            X = X.select_dtypes(include=[np.number])
            X_scaled = StandardScaler().fit_transform(X)

            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

            if st.button("🚀 Train Random Forest Model"):
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid = GridSearchCV(model, param_grid, cv=cv)
                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                # Accuracy
                acc = accuracy_score(y_test, y_pred)
                st.success("✅ Model Trained")
                st.write("### ✅ Total Model Accuracy", acc)
                st.write("### 📈 Best Parameters", grid.best_params_)

                # Classification Report
                st.text("### 📋 Classification Report")
                st.text(classification_report(y_test, y_pred))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("🔍 Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Feature Importance
                st.write("### 💡 Feature Importance")
                feat_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig2, ax2 = plt.subplots()
                sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax2)
                st.pyplot(fig2)

                # App-wise Risk (if column exists)
                if 'app_name' in df.columns:
                    df_temp = df.copy()
                    df_temp['prediction'] = best_model.predict(StandardScaler().fit_transform(df_temp[X.columns]))
                    non_prod_apps = df_temp[df_temp['prediction'] == 0]
                    prod_apps = df_temp[df_temp['prediction'] == 1]
                    harmful_apps = non_prod_apps['app_name'].value_counts()
                    safe_apps = prod_apps['app_name'].value_counts()

                    st.write("### 🔥 High Risk Apps (Predicted as Non-Productive)")
                    if not harmful_apps.empty:
                        st.bar_chart(harmful_apps)
                        for app, count in harmful_apps.items():
                            if count > 3:
                                st.warning(f"🚨 Alert: Excessive use of '{app}' detected. May harm sleep/health.")
                    else:
                        st.info("✅ No high-risk apps detected based on the current model predictions.")

                    st.write("### ✅ No-Risk Apps (Predicted as Productive)")
                    if not safe_apps.empty:
                        st.bar_chart(safe_apps)
                    else:
                        st.info("No low-risk apps predicted.")

                # Download Predictions
                predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                csv = predictions_df.to_csv(index=False)
                st.download_button("📥 Download Predictions CSV", csv, file_name='predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
