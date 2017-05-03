import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.metrics import roc_auc_score

from binning import TreeBinner

conversion_dataset = np.loadtxt('datasets/conversion.csv', delimiter = ',')

X = conversion_dataset[:, :-1]
y = conversion_dataset[:, -1]

# One hot encode the categorial features
ohe = OneHotEncoder(sparse = False)
X = np.hstack((ohe.fit_transform(X[:, :-2]), X[:, -2:]))

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print 'INITIAL - Dataset shape:', X_train.shape

# Train model before any transformations.
params_to_tune = {'C' : [0.1, 0.5, 1.0, 2.0, 10.0]}
lr = LogisticRegression(penalty='l1')
clf = GridSearchCV(lr, params_to_tune)
clf.fit(X_train, y_train)
tr_score = clf.score(X_train, y_train)
te_score = clf.score(X_test, y_test)
print 'RAW - Training Accuracy:', tr_score
print 'RAW - Testing Accuracty:', te_score
print 'RAW - Training AUC:', roc_auc_score(y_train, clf.predict(X_train))
print 'RAW - Testing AUC:', roc_auc_score(y_test, clf.predict(X_test))
print 'RAW - L1 C:', clf.best_estimator_.C

# Bin using tree
X_train_binned = X_train
X_test_binned = X_test
treeBin = TreeBinner(max_depth=4)

# First continuous column
tr_col_binned = treeBin.fit_transform(X_train[:, -2], y_train)
te_col_binned = treeBin.transform(X_test[:, -2])
X_train_binned = np.hstack((X_train_binned, tr_col_binned))
X_test_binned = np.hstack((X_test_binned, te_col_binned))

# Second continuous column
tr_col_binned = treeBin.fit_transform(X_train[:, -1], y_train)
te_col_binned = treeBin.transform(X_test[:, -1])
X_train_binned = np.hstack((X_train_binned, tr_col_binned))
X_test_binned = np.hstack((X_test_binned, te_col_binned))
print 'DONE BINNING - Dataset shape:', X_train_binned.shape

# Add interaction terms and basis expansion
poly = PolynomialFeatures(interaction_only = True)
X_train_binned = poly.fit_transform(X_train_binned)
X_test_binned = poly.fit_transform(X_test_binned)
print 'DONE ADDING TERMS - Dataset shape:', X_train_binned.shape

# Logistic Regression transformed data
params_to_tune = {'C' : [0.1, 0.5, 1.0, 2.0]}
lr= LogisticRegression(penalty='l1')
clf = GridSearchCV(lr, params_to_tune)
clf.fit(X_train_binned, y_train)
tr_score = clf.score(X_train_binned, y_train)
te_score = clf.score(X_test_binned, y_test)
print 'TRANSFORMED - Training Accuracy:', tr_score
print 'TRANSFORMED - Testing Accuracty:', te_score
print 'TRANSFORMED - Training AUC:', roc_auc_score(y_train, clf.predict(X_train_binned))
print 'TRANSFORMED - Testing AUC:', roc_auc_score(y_test, clf.predict(X_test_binned))
print 'TRANSFORMED - L1 C:', clf.best_estimator_.C
