"""
Task 1: Credit Scoring Model (German Credit Data)
Run: python credit_scoring_german.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

# --- Step 1: Load dataset (local file required) ---
col_names = [
    "Status_checking_account", "Duration_months", "Credit_history", "Purpose", "Credit_amount",
    "Savings_account_bonds", "Present_employment_since", "Installment_rate", "Personal_status_sex",
    "Other_debtors_guarantors", "Present_residence_since", "Property", "Age", "Other_installment_plans",
    "Housing", "Number_existing_credits", "Job", "Number_people_liable", "Telephone", "Foreign_worker", "Target"
]

df = pd.read_csv("german.data", delim_whitespace=True, header=None, names=col_names)

# Map target: 1 = good, 2 = bad
df['Target'] = df['Target'].map({1: 1, 2: 0})

# --- Step 2: Features and target ---
X = df.drop(columns=['Target'])
y = df['Target']

numeric_cols = ["Duration_months", "Credit_amount", "Installment_rate", 
                "Present_residence_since", "Age", "Number_existing_credits", "Number_people_liable"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# --- Step 3: Preprocessing ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# --- Step 4: Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Step 5: Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for name, model in models.items():
    print(f"\n=== {name} ===")
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("Classification Report:\n", classification_report(y_test, preds))

    joblib.dump(clf, f"{name.replace(' ', '_').lower()}_german_credit.joblib")
    print(f"Saved model as {name.replace(' ', '_').lower()}_german_credit.joblib")
