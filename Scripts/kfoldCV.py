"""
A function that can be used for k-fold Cross-Validation
Input are the dependent and independent variables, and the number k.
Output is the OOS R^2 and the OOS MAE
"""

import numpy as np
from linearmodels import PanelOLS
from sklearn.model_selection import KFold

def kfoldfun(y,X,k):
    rng = np.random.RandomState(seed=12345)
    s = 100
    seeds = np.arange(s)
    tot_error = 0
    rng.shuffle(seeds)
    rsqtot = 0
    for seed in seeds:
        cv = KFold(n_splits=k, shuffle=True,random_state = seed)
        for train_index, valid_index in cv.split(X, y):
            mod = PanelOLS(y.iloc[train_index], X.iloc[train_index], entity_effects = True)
            res = mod.fit(cov_type='clustered')
            pred = mod.predict(res.params,exog = X.iloc[valid_index])
            rsq = 1 - (((y.iloc[valid_index].to_numpy()-pred.to_numpy().transpose())**2).sum())/(((y.iloc[valid_index].to_numpy() - y.iloc[valid_index].to_numpy().mean())**2).sum())
            MSPE = np.abs((y.iloc[valid_index].to_numpy()-pred.to_numpy().transpose())).mean()
            tot_error = tot_error + MSPE
            rsqtot = rsqtot + rsq
    print("Mean Absolute Error:")
    print(tot_error/(s*k))
    print("OOS R^2")
    print(rsqtot/(s*k))
