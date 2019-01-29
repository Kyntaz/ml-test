# Pedro Quintas, 83546
# Gonçalo Gaspar, 83471
# Grupo 95

import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def features(X):
    
    F = np.zeros((len(X),45))
    for x in range(0,len(X)):
        F[x, 0] = len(X[x])
        F[x, 1] = sum([1 if c in "aeiou"  else 0 for c in X[x]])
        F[x, 2] = ord(X[x][1])
        F[x, 3] = ord(X[x][2])
        F[x, 4] = ord(X[x][0])

    return F

def mytraining(f,Y):
    # List of Potential Models.
    hyp = [
        tree.DecisionTreeClassifier()
    ]
    
    clf = mytrainingaux(f,Y,hyp)
    clf.fit(f,Y)
    return clf
    
def mytrainingaux(f,Y,par):
    # Use cross validation to choose the best classifier from a list.
    score = 0
    clf = None
    n = 0
    sel=0
    for c in par:
        n+=1
        ts = cross_val_score(c,f, y=Y, scoring="f1").mean();
        if ts > score:
            score = ts
            clf = c
            sel=n
    #print(sel)
    #print(score)
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f)
    return Ypred
