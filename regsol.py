# Pedro Quintas, 83546
# Gonçalo Gaspar, 83471
# Grupo 95

import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):
    hyp1 = [
        KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1),
        KernelRidge(alpha=1.0, kernel="polynomial", coef0=5),
    ]
    
    hyp2 = [
        tree.DecisionTreeRegressor()
    ]
    
    reg = mytrainingaux(X,Y,hyp1)
    reg = reg.fit(X,Y)
    return reg
    
def mytrainingaux(X,Y,par):
    # Use cross validation to choose the best regressor from a list.
    score = 1000
    reg = None
    n = 0
    sel=0
    for c in par:
        n+=1
        ts = -cross_val_score(c, X, y=Y, cv=5, scoring="neg_mean_squared_error").mean()
        if ts < score:
            score = ts
            reg = c
            sel=n
    #print(sel)
    #print(score)
    return reg if reg != None else par[0]

def myprediction(X,reg):
    Ypred = reg.predict(X)
    return Ypred
