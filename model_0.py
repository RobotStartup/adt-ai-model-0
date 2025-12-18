import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load NEW data
df = pd.read_csv("ai_model_sales.csv")

# Feature Engineering
X = pd.DataFrame()
X['service_history'] = df['service_history']
X['industry_revenue'] = df['industry_revenue']
X['number_of_locations'] = df['number_of_locations']
X['days_since_disconnect'] = df['days_since_disconnect']
X['historical_rmr'] = df['historical_rmr']
X['multiple_locations'] = df['multiple_locations']

X['has_panel'] = df['purchase_history'].str.contains('panel', na=False).astype(int)
X['has_cameras'] = df['purchase_history'].str.contains('cameras', na=False).astype(int)
X['has_monitoring'] = df['purchase_history'].str.contains('monitoring', na=False).astype(int)
X['has_fire'] = df['purchase_history'].str.contains('fire', na=False).astype(int)
X['has_access_control'] = df['purchase_history'].str.contains('access_control', na=False).astype(int)
X['no_services'] = (df['purchase_history'] == 'none').astype(int)
X['service_count'] = X[['has_panel','has_cameras','has_monitoring','has_fire','has_access_control']].sum(axis=1)
X['is_smart_panel'] = (df['equipment_service type'] == 'smart_panel').astype(int)
X['recency_score'] = 500 - X['days_since_disconnect'].clip(upper=500)
X['value_score'] = X['historical_rmr'] * X['service_history']

y = df["conversion"]

# Scale & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

print("\nFeature Importance:")
for name, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:25s}: {imp:.3f}")