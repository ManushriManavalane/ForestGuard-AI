import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Ensure folders exist
if not os.path.exists('models'):
    os.makedirs('models')

print("[*] Loading UCI Forest Fires dataset...")
df = pd.read_csv('data/forestfires.csv')
print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")

# Feature Engineering
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map   = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

df['month_num'] = df['month'].map(month_map).fillna(6)
df['day_num']   = df['day'].map(day_map).fillna(3)

# Fire spread label: area > 0.5 means actual spread
df['spread'] = (df['area'] > 0.5).astype(int)

# Features: temp, RH (humidity), wind, fire weather indices, month, day
features = ['temp', 'RH', 'wind', 'FFMC', 'DMC', 'DC', 'ISI', 'month_num', 'day_num']
X = df[features]
y = df['spread']

print(f"\n[*] Class distribution:\n{y.value_counts().to_dict()}")
print(f"    Spread rate: {y.mean()*100:.1f}%")

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Model 1: Gradient Boosting (primary)
print("\n[*] Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
gb_acc   = accuracy_score(y_test, gb_preds)
print(f"    [OK] GradientBoosting Accuracy: {gb_acc*100:.2f}%")
print(classification_report(y_test, gb_preds, target_names=['No Spread', 'Spread']))

# Model 2: Random Forest (ensemble support)
print("[*] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"    [OK] RandomForest Accuracy: {rf_acc*100:.2f}%")

# Save models
joblib.dump(gb_model, 'models/fire_model.pkl')
joblib.dump(rf_model, 'models/fire_rf_model.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(features, f)

print("\n[DONE] Models saved:")
print("  models/fire_model.pkl        (GradientBoosting - primary)")
print("  models/fire_rf_model.pkl     (RandomForest - ensemble)")
print("  models/feature_names.json    (feature list)")
print(f"\n[BEST] GradientBoosting: {gb_acc*100:.2f}%")