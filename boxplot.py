# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:17:45 2020

@author: Pierre
"""

import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
X = pd.read_csv("data/data.csv").values[: ,1:].astype(float)
Y = pd.read_csv("data/labels.csv").values[:, 1:]

plt.figure()
prad = np.where(Y == "PRAD")[0]
noprad = np.where(Y != "PRAD")[0]
plt.boxplot((X[prad, 203],X[noprad, 203]), labels = ("PRAD", "Others"))
plt.savefig("outputs/boxplot203.png")

plt.figure()
luad = np.where(Y == "LUAD")[0]
noluad = np.where(Y != "LUAD")[0]
plt.boxplot((X[luad, 15898],X[noluad, 15898]), labels = ("LUAD", "Others"))
plt.savefig("outputs/boxplot15989.png")
