import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Load data
df = pd.read_csv("ai_model_sales.csv")

# 2. Split features and target
X = df.drop(columns=["company", "conversion"])  # drop company name too
y = df["conversion"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split (stratify keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train model with class balancing
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # KEY: handles imbalanced data
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))