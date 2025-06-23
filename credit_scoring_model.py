# codealpha - Credit Scoring Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Simulated Dataset (balanced)
data = {
    'income': [40000, 60000, 50000, 80000, 30000, 45000, 35000, 70000],
    'debts': [10000, 5000, 7000, 2000, 12000, 9000, 15000, 4000],
    'payment_history': [1, 0, 1, 1, 0, 0, 0, 1],  # 1 = Good, 0 = Bad
    'creditworthy': [1, 1, 1, 1, 0, 0, 0, 1]      # Target
}
df = pd.DataFrame(data)

# Step 2: Features & Target
X = df[['income', 'debts', 'payment_history']]
y = df['creditworthy']

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Output Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: ROC-AUC Score (safely check class count)
if len(model.classes_) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
else:
    print("ROC-AUC Score cannot be calculated — only one class present.")
