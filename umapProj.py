# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:28:08 2019

@author: Pierre
"""

import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
X = pd.read_csv("data/data.csv").values[: ,1:]
Y = pd.read_csv("data/labels.csv").values[:, 1:]
# Encode labels
encoder = LabelEncoder().fit(Y)
Y = encoder.transform(Y)

# 2D UMAP plot
plots = []
plt.figure()
umap2D = umap.UMAP().fit_transform(X)
for i in range(5):
    inds = np.where(Y==i)[0]
    plots.append(plt.scatter(umap2D[inds, 0], umap2D[inds, 1]))
plt.legend(plots,encoder.classes_[np.arange(0,5)])
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.title("Décomposition UMAP du jeu de données")
plt.savefig("outputs/2D_UMAP.png")

# 2D PCA plot
plots = []
plt.figure()
umap2D = PCA().fit_transform(X)
for i in range(5):
    inds = np.where(Y==i)[0]
    plots.append(plt.scatter(umap2D[inds, 0], umap2D[inds, 1]))
plt.legend(plots,encoder.classes_[np.arange(0,5)])
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.title("Décomposition PCA du jeu de données")
plt.savefig("outputs/2D_PCA.png")
