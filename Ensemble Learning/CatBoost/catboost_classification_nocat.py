import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import catboost as cb

# 一共有约 500 万条记录，我使用了 1% 的数据：5 万行记录
# data = pd.read_csv("flight-delays/flights.csv")
# data = data.sample(frac=0.1, random_state=10)  # 500->50
# data = data.sample(frac=0.1, random_state=10)  # 50->5
# data.to_csv("flight-delays/min_flights.csv")

# 读取 5 万行记录
data = pd.read_csv("flight-delays/min_flights.csv")
print(data.shape)  # (58191, 31)

data = data[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT",
             "ORIGIN_AIRPORT", "AIR_TIME", "DEPARTURE_TIME", "DISTANCE", "ARRIVAL_DELAY"]]
data.dropna(inplace=True)

data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"] > 10) * 1

cols = ["AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes + 1

train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
                                                random_state=10, test_size=0.25)

cat_features_index = [0, 1, 2, 3, 4, 5, 6]


def auc(m, train, test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))


# 调参，用网格搜索调出最优参数
params = {'depth': [4, 7, 10],
          'learning_rate': [0.03, 0.1, 0.15],
          'l2_leaf_reg': [1, 4, 9],
          'iterations': [300, 500]}
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv=3)
cb_model.fit(train, y_train)
# 查看最佳分数
print(cb_model.best_score_)  # 0.7088001891107445
# 查看最佳参数
print(cb_model.best_params_)  # {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 9, 'learning_rate': 0.15}

# With Categorical features，用最优参数拟合数据
clf = cb.CatBoostClassifier(eval_metric="AUC", depth=4, iterations=500, l2_leaf_reg=9,
                            learning_rate=0.15)

clf.fit(train, y_train)

print(auc(clf, train, test))  # (0.7809684655761157, 0.7104617034553192)
