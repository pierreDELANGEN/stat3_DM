# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:28:08 2019

@author: Pierre
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostError
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import umap
import xgboost
from sklearn.decomposition import PCA


nDims = 20


def fitModel(X, Y):
    try:
        model = CatBoostClassifier(iterations=200, task_type="GPU")
        model.fit(X, Y, silent=True)
    except CatBoostError:   # No gpu
        model = CatBoostClassifier(iterations=200)
        model.fit(X, Y, silent=True)
    return model


# Load dataset
X = pd.read_csv("data/data.csv").values[:, 1:]
Y = pd.read_csv("data/labels.csv").values[:, 1:]
# Encode labels
encoder = LabelEncoder().fit(Y)
Y = encoder.transform(Y)

# XGBoost feature selection test ---------------------------------------------
# Prepare CV predictions array
predictions = np.empty((len(X), 5))
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# CV
for train, test in kfold.split(X, Y):
    model = xgboost.XGBClassifier()
    model.fit(X[train], Y[train])
    importances = model.feature_importances_
    featureSet = np.argsort(importances)[::-1][:nDims]
    X_train = X[train][:, featureSet]
    X_test = X[test][:, featureSet]
    model = fitModel(X_train, Y[train])
    predictions[test] = model.predict_proba(X_test)
print("Accuracy with XGBoost feature selection :",
      accuracy_score(Y, np.argmax(predictions, axis=1)))


# PCA feature selection test -------------------------------------------------
# Prepare CV predictions array
predictions = np.empty((len(X), 5))
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# CV
for train, test in kfold.split(X, Y):
    pca = PCA().fit(X[train])
    pca_importance = np.sum(abs( pca.components_ ), axis=0)
    featureSet = np.argsort(pca_importance)[::-1][:nDims]
    X_train = X[train][:, featureSet]
    X_test = X[test][:, featureSet]
    model = fitModel(X_train, Y[train])
    predictions[test] = model.predict_proba(X_test)
print("Accuracy with PCA feature selection :",
      accuracy_score(Y, np.argmax(predictions, axis=1)))


# PCA dimensionnality reduction test -----------------------------------------
# Prepare CV predictions array
predictions = np.empty((len(X), 5))
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# CV
for train, test in kfold.split(X, Y):
    pca = PCA().fit(X[train])
    X_train = pca.transform(X[train])[:, :nDims]
    X_test = pca.transform(X[test])[:, :nDims]
    model = fitModel(X_train, Y[train])
    predictions[test] = model.predict_proba(X_test)
print("Accuracy with PCA dimension reduction :",
      accuracy_score(Y, np.argmax(predictions, axis=1)))


# UMAP dimensionnality reduction test -----------------------------------------
# Prepare CV predictions array
predictions = np.empty((len(X), 5))
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# CV
for train, test in kfold.split(X, Y):
    dimReduction = umap.UMAP(n_components=nDims).fit(X[train])
    X_train = dimReduction.transform(X[train])
    X_test = dimReduction.transform(X[test])
    model = fitModel(X_train, Y[train])
    predictions[test] = model.predict_proba(X_test)
print("Accuracy with UMAP dimension reduction :",
      accuracy_score(Y, np.argmax(predictions, axis=1)))
