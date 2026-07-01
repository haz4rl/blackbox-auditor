import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import pickle

print("📥 Fetching real UCI German Credit Dataset from OpenML...")
# Download the real global benchmarking credit dataset
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='pandas')
df = credit_data.frame

print("⚙️ Transforming features into production-ready schemas...")
# Map the target label to standard binary logic (1: Good risk, 0: Bad risk)
df['target'] = df['class'].map({'good': 1, 'bad': 0})

# Standardize feature frames using key predictive variables from the dataset
processed_df = pd.DataFrame({
    'Age': df['age'].astype(float),
    'Credit_Amount': df['credit_amount'].astype(float),
    'Duration_Months': df['duration'].astype(float),
    # Extract structural gender demographic from string format personal_status
    'Gender_Male': df['personal_status'].apply(lambda x: 0 if 'female' in str(x) else 1)
})

X = processed_df
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Black-Box Ensemble Classifier Pipeline...")
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

print("💾 Exporting real-world artifacts and serialized weights...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("✅ Initialization complete! Real-world system dependencies saved successfully.")
