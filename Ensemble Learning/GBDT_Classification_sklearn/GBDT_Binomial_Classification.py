import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=5, subsample=1
                                 , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                 , init=None, random_state=None, max_features=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False
                                 )