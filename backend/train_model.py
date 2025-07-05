import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'AI Hakathon.csv'))

# Yahan each property type ko Label kr raha as 0 1 2 3 4 
le = LabelEncoder()
df['PROPERTYTYPE_ENC'] = le.fit_transform(df['PROPERTYTYPE'])

# Features and target
features = ['LOTSIZEACRES', 'YEARBUILT', 'BEDS', 'BATHS', 'PROPERTYTYPE_ENC']
target = 'SALESPRICE'

x = df[features]
y = df[target]

# Splitting into test/train and training model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
model_sale = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=5
)
model_sale.fit(x_train, y_train)

# Save model and encoder
joblib.dump(model_sale, "SALESPRICE_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Metrics
y_pred = model_sale.predict(x_test)
print(f"SALESPRICE Model -> MAE: {mean_absolute_error(y_test, y_pred):.2f}, R2: {r2_score(y_test, y_pred):.2f}")