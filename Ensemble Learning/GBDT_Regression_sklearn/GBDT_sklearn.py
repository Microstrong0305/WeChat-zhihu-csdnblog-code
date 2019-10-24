import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=5, subsample=1
                                 , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                 , init=None, random_state=None, max_features=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False
                                 )
train_feat = np.array([[1, 5, 20],
                       [2, 7, 30],
                       [3, 21, 70],
                       [4, 30, 60],
                       ])
train_id = np.array([[1.1], [1.3], [1.7], [1.8]]).ravel()
test_feat = np.array([[5, 25, 65]])
test_id = np.array([[1.6]])
print(train_feat.shape, train_id.shape, test_feat.shape, test_id.shape)
gbdt.fit(train_feat, train_id)
pred = gbdt.predict(test_feat)
total_err = 0
for i in range(pred.shape[0]):
    print(pred[i], test_id[i])
    err = (pred[i] - test_id[i]) / test_id[i]
    total_err += err * err
print(total_err / pred.shape[0])
