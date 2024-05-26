from sklearn.datasets import fetch_openml
import matplotlib
from sklearn.svm import LinearSVC, SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#fetch the mnist datset from openml library
mnist = fetch_openml('mnist_784', version=1)
#images
X = mnist["data"]
#labels
y = mnist["target"].astype(np.int64)

# Select only digits 2, 3, 8, and 9
classes = [2, 3, 8, 9]
mask = np.isin(y, classes)
X, y = X[mask], y[mask]

# Reset index to align indices correctly
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Normalizing, flattinging the graysacale image
X = X / 255.0

#region linear SVM

#20000 training and 4000 test data 4/24 test size = 0.1667 , random value arbitraarry
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1667, random_state=42)
#Regularization: Hyperparameter grid using 4 different c values 
param_grid = {'C': [0.1, 1, 10, 100]}
#linear support vector machine
linear_svc = LinearSVC(max_iter=10000)
# 5 fold cross vlaidation
grid_search = GridSearchCV(linear_svc, param_grid, cv=5)
#train the model
grid_search.fit(X_train, y_train)



print("Best parameters for Linear SVM:", grid_search.best_params_)
print("Training accuracy for Linear SVM:", grid_search.best_score_)

# Evaluate on the test set
best_linear_model = grid_search.best_estimator_
y_test_pred = best_linear_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test accuracy for Linear SVM:", test_accuracy)

#endregion


#region non-linear SVM Train a Non-Linear SVM with RBF kernel

# Reduce the dimensionality
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train_pca, y_test_pca, train_indices_pca, test_indices_pca = train_test_split(
    X_pca, y, np.arange(len(y)), test_size=0.1667, random_state=42)

# Hyperparameter grid for RBF kernel
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
rbf_svc = SVC(kernel='rbf')
grid_search_rbf = GridSearchCV(rbf_svc, param_grid, cv=5, n_jobs=6)
grid_search_rbf.fit(X_train_pca, y_train_pca)

print("Best parameters for RBF kernel:", grid_search_rbf.best_params_)
print("Training accuracy for RBF kernel:", grid_search_rbf.best_score_)

# Evaluate on the test set for RBF kernel
best_model_rbf = grid_search_rbf.best_estimator_
y_test_pred_rbf = best_model_rbf.predict(X_test_pca)
test_accuracy_rbf = accuracy_score(y_test_pca, y_test_pred_rbf)
print("Test accuracy for RBF kernel:", test_accuracy_rbf)

#endregion

# Get the support vectors indices in the original dataset
support_indices = best_model_rbf.support_

# Map the support vector indices to the original training set indices
support_indices_original = train_indices_pca[support_indices]

# Convert X to a NumPy array if it's not already
X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else X

plt.figure(figsize=(12, 12))
for i, index in enumerate(support_indices_original[:64]):  # Limiting to first 64 support vectors for visualization
    plt.subplot(8, 8, i + 1)
    plt.imshow(X_array[index].reshape(28, 28), cmap="gray")
    plt.axis('off')
    plt.title(f"Label: {y.iloc[index]}")
plt.suptitle("Support Vectors")
plt.show()