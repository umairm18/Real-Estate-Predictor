import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, '..', 'AI Hakathon.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'SALESPRICE_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

df = pd.read_csv(CSV_PATH)

if os.path.exists(ENCODER_PATH):
    le = joblib.load(ENCODER_PATH)
else:
    le = LabelEncoder()
    df['PROPERTYTYPE_ENC'] = le.fit_transform(df['PROPERTYTYPE'])
    joblib.dump(le, ENCODER_PATH)

if 'PROPERTYTYPE_ENC' not in df.columns:
    df['PROPERTYTYPE_ENC'] = le.transform(df['PROPERTYTYPE'])

features = ['LOTSIZEACRES', 'YEARBUILT', 'BEDS', 'BATHS', 'PROPERTYTYPE_ENC']
target = 'SALESPRICE'

X = df[features]
y = df[target]

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=50)

# Existing Model if Available
if os.path.exists(MODEL_PATH):
    old_model = joblib.load(MODEL_PATH)
    print("Loaded existing model. Fine-tuning...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=5
    )
    model.fit(
        x_train,
        y_train,
        init_model=old_model.booster_  # continue from existing
    )
else:
    print("🚀 No existing model found. Training from scratch...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=5
    )
    model.fit(x_train, y_train)

# Save Updated Model
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# Evaluation
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Fine-tuned Model Performance:")
print(f"→ MAE: {mae:.2f}")
print(f"→ R² Score: {r2:.2f}")