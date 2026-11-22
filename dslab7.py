
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Wine Dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("Wine Quality Classification")
print("=" * 65)
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", list(wine.target_names))
print("Class counts:", np.bincount(y).tolist())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Define Models
models = {
    "Gaussian Naive Bayes": GaussianNB(),

    "k-Nearest Neighbours (k=5)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Decision Tree (depth=3)": DecisionTreeClassifier(
        max_depth=3, random_state=42
    )
}

# 5-Fold Cross Validation
kfold = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

summary = []

for name, model in models.items():
    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    cv = cross_val_score(model, X, y, cv=kfold)

    summary.append([
        name, train_acc, test_acc, cv.mean(), cv.std()
    ])

# Model Comparison
table = pd.DataFrame(
    summary,
    columns=["Model", "Train Acc", "Test Acc", "CV Mean", "CV Std"]
)

print("\nModel Comparison:")
print(table.round(4).to_string(index=False))

# Select Best Model
best = table.sort_values(
    ["CV Mean", "Test Acc"],
    ascending=False
).iloc[0]["Model"]

print("\nSelected Model:", best)

pred = models[best].predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, pred, target_names=wine.target_names
))

# Effect of k in KNN
print("\nEffect of k on KNN:")
for k in [1, 3, 5, 7, 11, 15]:
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    score = cross_val_score(
        knn, X, y, cv=kfold
    ).mean()

    print(f"k = {k:>2} -> {score:.4f}")

# Decision Tree Rules
print("\nDecision Tree Rules:")

print(export_text(
    models["Decision Tree (depth=3)"],
    feature_names=list(X.columns)
))
