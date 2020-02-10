# Catboost for Avito Demand Prediction Challenge
# https://www.kaggle.com/c/avito-demand-prediction
# Reference: https://www.kaggle.com/nicapotato/simple-catboost

import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

print("\nData Load Stage")
training = pd.read_csv('Avito_Demand_Prediction_Challenge/train.csv', index_col="item_id",
                       parse_dates=["activation_date"])  # .sample(1000)
traindex = training.index
testing = pd.read_csv('Avito_Demand_Prediction_Challenge/test.csv', index_col="item_id",
                      parse_dates=["activation_date"])  # .sample(1000)
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"] + 0.001)
df["price"].fillna(-999, inplace=True)
df["image_top_1"].fillna(-999, inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type",
               "image_top_1"]
messy_categorical = ["param_1", "param_2", "param_3", "title", "description"]  # Need to find better technique for these
print("Encoding :", categorical + messy_categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

print("\nCatboost Modeling Stage")
X = df.loc[traindex, :].copy()
print("Training Set shape", X.shape)
test = df.loc[testdex, :].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()

# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)


# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


categorical_features_pos = column_index(X, categorical + messy_categorical)

# Train Model
print("Train CatBoost Decision Tree")
modelstart = time.time()
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='RMSE',
                             random_seed=23,
                             bagging_temperature=0.2,
                             od_type='Iter',
                             metric_period=75,
                             od_wait=100)
cb_model.fit(X_train, y_train,
             eval_set=(X_valid, y_valid),
             cat_features=categorical_features_pos,
             use_best_model=True,
             verbose=True)

# # Feature Importance
# fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': X.columns})
# fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
# _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
# plt.savefig('catboost_feature_importance.png')

print("Model Evaluation Stage")
print(cb_model.get_params())
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, cb_model.predict(X_valid))))
catpred = cb_model.predict(test)
catsub = pd.DataFrame(catpred, columns=["deal_probability"], index=testdex)
catsub['deal_probability'].clip(0.0, 1.0, inplace=True)
catsub.to_csv("catsub.csv", index=True, header=True)  # Between 0 and 1
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))
