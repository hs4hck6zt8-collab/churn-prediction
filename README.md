# Churn Prediction - Customer Retention Model

Предсказание оттока клиентов телеком-компании с объяснением причин  на уровне каждого клиента.

## Результаты модели

| Метрика        | Значение |
|----------------|----------|
| ROC-AUC        | 0.8298   |
| Recall (churn) | 0.75     |
| F1 (churn)     | 0.61     |
| Threshold      | 0.23     |

## Архитектура

Raw Data → EDA → Feature Engineering → sklearn Pipeline → Stacking Ensemble → Threshold Tuning → FastAPI

**Стекинг:** LightGBM + CatBoost → LogisticRegression (meta)  
**Объяснимость:** SHAP TreeExplainer  
**API:** FastAPI /predict + /explain  

## Ключевые признаки

1. **Contract** - month-to-month контракт резко повышает churn
2. **tenure** - новые клиенты (< 12 мес) уходят чаще
3. **MonthlyCharges** - высокий чек коррелирует с оттоком
4. **monthly_charges_per_service** - engineered feature
5. **OnlineSecurity / TechSupport** - отсутствие повышает churn

## Структура проекта

churn-prediction/  
├──data/  
│ ├──raw/    #Исходный датасет (нет в git)  
│ └──processed/ #После feature engineering
├──notebooks/  
│ ├──01_eda.ipynb  
│ ├──02_features.ipynb  
│ └──03_shap.ipynb  
├──src/  
│ ├──train.py #Обучение модели  
│ └──api.py #FastAPI сервис  
├──models/  #Сохраненная модель  
├──reports/ #SHAP графики  
├──Dockerfile  
└──requirements.txt  

## Быстрый старт

```bash
# Установка
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Обучение
python src/train.py

# Запуск API
uvicorn src.api:app --reload
# Документация: http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
     "Contract": "Month-to-month",
     "tenure": 2,
     "MonthlyCharges": 75.5,
     ...
  }'
```

### Ответ

```json
{
  "churn_probability": 0.8341,
  "churn_prediction": 1,
  "threshold_used": 0.23
}
```

## Датасет

[Telco Customer Churn — Kaggle]([https://www.kaggle.com/datasets/blastchar/telco-customer-churn])

7043 клиента, 21 признаков, 26.5% churn rate.

## License

MIT License
