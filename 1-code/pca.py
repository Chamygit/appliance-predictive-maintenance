# -----------------------------------------------------------
# Script used for doing a PCA analysis on the features dataset
#
# (C) 2022 Tiago Fonseca, Porto, Portugal
# This work was supported by project SMART-PDM https://smart-pdm.eu/
# Released under GNU Public License (GPL)
# email calof@isep.ipp.pt
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt  # must import pyplot submodule, not top-level matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file = "X.csv"  # features dataset file (fridges or washing machines)
df = pd.read_csv(file, sep=",")

X_a = df.iloc[:, 1:-2]
y = df.iloc[:, -1].values
features = X_a.columns
X = X_a.values

# Standardize before PCA
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
target = pd.Series(y, name="target")
result_df = pd.concat([df_pca, target], axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("First Principal Component", fontsize=15)
ax.set_ylabel("Second Principal Component", fontsize=15)

# ── FOR REFRIGERATORS: uncomment the following 3 lines ──────────────────────
# ax.set_title("PCA (2 PCs) — Refrigerator Dataset", fontsize=20)
# targets = [0, 1]
# colors = ["g", "r"]

# ── FOR WASHING MACHINES: uncomment the following 3 lines ───────────────────
ax.set_title("PCA (2 PCs) — Washing Machine Dataset", fontsize=20)
targets = [0, 1, 2, 3]
colors = ["g", "r", "b", "m"]

for target_class, color in zip(targets, colors):
    mask = y == target_class
    ax.scatter(result_df.loc[mask, "PC1"], result_df.loc[mask, "PC2"], c=color, s=50)

ax.legend(targets)
ax.grid()
plt.show()

print("Variance of each component:", pca.explained_variance_ratio_)
print("\nTotal Variance Explained:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")
