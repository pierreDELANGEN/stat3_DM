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


nGenes = 10


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

# XGBoost feature selection ---------------------------------------------------
for c in range(5):  #For each cancer
    # Prepare CV predictions array
    predictions = np.empty((len(X), 2))
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    newY = (Y == c).astype(int)       # 1 = studied cancer, 0 = others
    geneInfluence = np.zeros(X.shape[1])    # Store gene influence for each CV stage
    # CV
    for train, test in kfold.split(X, Y):
        model = xgboost.XGBClassifier()
        model.fit(X[train], newY[train])
        importances = model.feature_importances_
        geneInfluence += importances
        featureSet = np.argsort(importances)[::-1][:nGenes]
        X_train = X[train][:, featureSet]
        X_test = X[test][:, featureSet]
        model = fitModel(X_train, newY[train])
        predictions[test] = model.predict_proba(X_test)
    print("Accuracy with %i genes on %s cancer :" %(nGenes, encoder.classes_[c]),
          accuracy_score(newY, np.argmax(predictions, axis=1)))
    
    pos = np.argsort(geneInfluence)[::-1]
    geneInfluence = geneInfluence[pos]/10.0
    outputTab = "Gene\tImportance\n"
    for i in range(len(geneInfluence)):
        outputTab += "gene_" + str(pos[i]) + "\t" + str(geneInfluence[i]) + "\n"
    f = open("outputs/%s_involved_genes.tsv" %(encoder.classes_[c]), "w")
    f.write(outputTab)
    f.close()