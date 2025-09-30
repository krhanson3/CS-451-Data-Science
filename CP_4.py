#CS 451/551
#Coding Practice Session 4
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#Feature Selection and Dimensionality Reduction 

#1
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

X = df[iris.feature_names]
y = df['species']

chi2_selector = SelectKBest(score_func=chi2, k='all') 
X_kbest = chi2_selector.fit_transform(X, y)

scores = chi2_selector.scores_
for feature, score in zip(iris.feature_names, scores):
    print(f"{feature}: {score:.4f}")

#2 Informaton Gain (Mutual Information) for Feature Selection
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target
mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    "Feature": iris.feature_names,
    "Mutual Information": mi_scores
}).sort_values(by="Mutual Information", ascending=False)

print(mi_df.to_string(index=False))

#3 Sequential Forward Selection (SFS)
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

knn = KNC(n_neighbors=5)

scores = []
for k in range(1, X.shape[1]):
    sfs_k = SFS(knn, n_features_to_select=k, direction="forward", scoring="accuracy", cv=5)
    sfs_k.fit(X, y)
    selected = X.columns[sfs_k.get_support()]
    acc = cross_val_score(knn, X[selected], y, cv=5, scoring="accuracy").mean()
    scores.append((k, list(selected), acc))

print("\nSFS:")
for step, feats, acc in scores:
    print(f"step {step}: seatures={feats}, accuracy={acc:.4f}")

#4 Random Forest for Feature Importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

clf = RFC(n_estimators=100, random_state=42)
clf.fit(X, y)

feature_importance = clf.feature_importances_

accuracy = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

plt.figure(figsize=(8, 5))
plt.bar(data.feature_names, feature_importance)
plt.title("RF feature importance")
plt.ylabel("score")
plt.xticks(rotation=45)
plt.show()

for feat, score in zip(data.feature_names, feature_importance):
    print(f"{feat}: {score:.4f}")

print(f"\nclassification accuracy: {accuracy:.4f}")


#5
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis", edgecolor="k", s=70)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2D) on Iris Dataset")
plt.colorbar(label="Species")
plt.show()

print("Explained variance ratio:", f'{pca.explained_variance_ratio_:.4f}')
print("Cumulative variance explained:", f'{pca.explained_variance_ratio_.cumsum():.4f}')


#6 Linear Discriminant Analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(8,6))
plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap="viridis", edgecolor="k", s=70)
plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.title("LDA on Iris Dataset")
plt.colorbar(label="Species")
plt.show()

#7 Feature Selection with Embedded Methods: Decision Trees

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

importances = tree.feature_importances_
for feat, score in zip(X.columns, importances):
    print(f"{feat}: {score:.4f}")

plt.figure(figsize=(8,5))
plt.bar(X.columns, importances)
plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=data.feature_names, class_names=data.target_names, filled=True, rounded=True)
plt.title("Decision Tree Classifier on Iris Dataset")
plt.show()

#8 Correlation-Based Feature Selection (CFS)
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["species"] = data.target

corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix, "\n")

target_corr = corr_matrix["species"].drop("species").abs().sort_values(ascending=False)
print("Features ranked by correlation with class label:")
print(target_corr, "\n")

feature_corr = corr_matrix.drop("species", axis=0).drop("species", axis=1)
print("Feature-to-feature correlation matrix:")
print(feature_corr)

#9 Lasso Regularization for Feature Selection
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

lasso = LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42, multi_class="multinomial")
lasso.fit(X, y)

coefficients = pd.DataFrame(lasso.coef_, columns=X.columns, index=data.target_names)
print("Lasso (L1) Coefficients per Class:")
print(coefficients, "\n")

importance = coefficients.abs().sum(axis=0).sort_values(ascending=False)
print("Features ranked by Lasso importance:")
print(importance)
