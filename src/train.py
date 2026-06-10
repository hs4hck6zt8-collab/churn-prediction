import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from jupyter_server import DEFAULT_STATIC_FILES_PATH
from plotly.graph_objs.indicator.gauge import threshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.compose import ColumnTransformer

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Пути
DATA_PATH = Path("data/processed/telco_featured.csv")
MODEL_PATH = Path("models/churn_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

# Загрузка данных
df = pd.read_csv(DATA_PATH)

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_segment'
]
NUM_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'monthly_charges_per_service', 'has_multiple_contracts'
]

FEATURES = CAT_COLS + NUM_COLS
TARGET = 'Churn'

X = df[FEATURES]
y = df[TARGET]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=-1))
])
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipe, CAT_COLS),
    ('num', num_pipe, NUM_COLS)
])

# Basic models
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                      num_leaves=40, random_state=42, verbose=-1)
catboost = CatBoostClassifier(iterations=500, learning_rate=0.05,
                              depth=6, random_state=42, verbose=0)

# Stacking
stacking = StackingClassifier(
    estimators=[('lgbm', lgbm), ('catboost', catboost)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=StratifiedKFold(n_splits=5),
    passthrough=False
)

# Final pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking)
])

# Обучение
print("Fitting model...")
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===Метрики===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Подбор порога
print("\n===Подбираем порог")
thresholds = np.arange(0.1, 0.85, 0.01)
best_thresh = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred_t = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    recall = (y_pred_t[y_test == 1] == 1).mean()
    print(f"Thresh={thresh:.2f}; F1={f1:.3f}; Recall={recall:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nЛучший порог: {best_thresh:.2f}; F1: {best_f1:.3f}")

# Финальные метрики
y_pred_best = (y_proba >= best_thresh).astype(int)
print("\n---Метрики с лучшим порогом---")
print(classification_report(y_test, y_pred_best))

# Saving
joblib.dump(
    {
        'model': model,
        'threshold': best_thresh
    },
    MODEL_PATH
)
print(f"Модель сохранена: {MODEL_PATH}")

