# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.8.13 ('dev')
#     language: python
#     name: python3
# ---

# # Classification with Tabular Data using H2OAutoML and Cleanlab
#

# This notebook is based on the following two tutorial notebooks.
#
# - [cleanlab/docs/source/tutorials/tabular.ipynb](https://github.com/cleanlab/cleanlab/blob/0dc384a4edfba31500e672b15026b781ea952f91/docs/source/tutorials/tabular.ipynb)
# - [h2o-tutorials/tutorials/sklearn-integration/H2OAutoML_as_sklearn_estimator.ipynb](https://github.com/h2oai/h2o-tutorials/blob/7c8fca34b2bf26870be71232ade52472a087f0ad/tutorials/sklearn-integration/H2OAutoML_as_sklearn_estimator.ipynb)
#

# In this tutorial, we will use `cleanlab` with `H2OAutoML` models to find potential label errors in the German Credit dataset. This dataset contains 1,000 individuals described by 20 features, each labeled as either "good" or "bad" credit risk. `cleanlab` automatically shortlists examples from this dataset that confuse our ML model; many of which are potential label errors (due to annotator mistakes), edge cases, and otherwise ambiguous examples.
#
# **Overview of what we'll do in this tutorial:**
#
# - Build a simple credit risk classifier with `H2OAutoML`.
#
# - Use this classifier to compute out-of-sample predicted probabilities, `pred_probs`, via cross validation.
#
# - Identify potential label errors in the data with `cleanlab`'s `find_label_issues` method.
#
# - Train a robust version of the same `H2OAutoML` model via `cleanlab`'s `CleanLearning` wrapper.
#
# **Data:** https://www.openml.org/d/31
#

# +
import random

import cleanlab
import h2o
import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from h2o.sklearn import H2OAutoMLClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# -

h2o.init()

# ## Install required dependencies
#

# You can use `conda` to install all packages required for this tutorial as follows:
#
# ```
# # !conda env update -n cleanlab-h2oautoml -f ./conda-env.yml
# ```
#

SEED = 123456
np.random.seed(SEED)
random.seed(SEED)

# ## Load and process the data
#

# We first load the data features and labels.
#

data = fetch_openml("credit-g")  # get the credit data from OpenML
X_raw = data.data  # features (pandas DataFrame)
y_raw = data.target  # labels (pandas Series)

# Next we preprocess the data. Here we apply one-hot encoding to features with categorical data, and standardize features with numeric data. We also perform label encoding on the labels - "bad" is encoded as 0 and "good" is encoded as 1.
#

# +
cat_features = X_raw.select_dtypes("category").columns
X_encoded = pd.get_dummies(X_raw, columns=cat_features, drop_first=True)

num_features = X_raw.select_dtypes("float64").columns
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])
X_scaled = X_scaled.to_numpy()

y = y_raw.map({"bad": 0, "good": 1})  # encode labels as integers
y = y.to_numpy()
# -

# Following proper ML practice, let's split our data into train and test sets.
#

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=SEED
)

# We again standardize the numeric features, this time fitting the scaling parameters solely on the training set.
#

# +
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


# -

# ## Select a classification model and compute out-of-sample predicted probabilities
#

# Here we use `H2OAutoML`, but you can choose _any_ suitable scikit-learn model for this tutorial.
#

# To identify label issues, `cleanlab` requires a probabilistic prediction from your model for every datapoint. However, these predictions will be _overfitted_ (and thus unreliable) for examples the model was previously trained on. `cleanlab` is intended to only be used with **out-of-sample** predicted probabilities, i.e., on examples held out from the model during the training.
#
# K-fold cross-validation is a straightforward way to produce out-of-sample predicted probabilities for every datapoint in the dataset by training K copies of our model on different data subsets and using each copy to predict on the subset of data it did not see during training. An additional benefit of cross-validation is that it provides a more reliable evaluation of our model than a single training/validation split. We can obtain cross-validated out-of-sample predicted probabilities from any classifier via a simple scikit-learn wrapper:
#

def getH2O():
    return H2OAutoMLClassifier(
        max_runtime_secs=10,
        # sort_metric="aucpr",
        verbosity="error",
    )


# `cleanlab` provides a wrapper class that can be easily applied to any scikit-learn compatible model. Once wrapped, the resulting model can still be used in the exact same manner, but it will now train more robustly if the data have noisy labels.
#

# Let's now train and evaluate the original `H2OAutoML` model.
#

og_h2o = getH2O()
og_h2o.fit(X_train, y_train)


# +
def eval_model(m, X, y):
    preds = m.predict(X)
    probas = m.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probas)
    return acc, auc


acc_og_og, auc_og_og = eval_model(og_h2o, X_test, y_test)
print(f"Test accuracy&auc of original H2OAutoML: {acc_og_og}, {auc_og_og}")
# -

# ## Train a more robust model from noisy labels
#

# The following operations take place when we train the `cleanlab`-wrapped model: The original model is trained in a cross-validated fashion to produce out-of-sample predicted probabilities. Then, these predicted probabilities are used to identify label issues, which are then removed from the dataset. Finally, the original model is trained on the remaining clean subset of the data once more.
#

# Use `frac_noise` to specify how much potentially erroneous noise should be automagically cleaned.

cl_h2o = CleanLearning(getH2O(), find_label_issues_kwargs=dict(frac_noise=0.5))

cl_h2o.fit(X_train, y_train)

# ## Dataset health summary

tr_data_health = cleanlab.dataset.health_summary(
    labels=y_train,
    pred_probs=cl_h2o.predict_proba(X_train),
    confident_joint=cl_h2o.confident_joint,
)

# Based on the given labels and out-of-sample predicted probabilities, `cleanlab` can quickly help us identify label issues in our dataset. Here we request that the indices of the identified label issues be sorted by `cleanlab`'s self-confidence score, which measures the quality of each given label via the probability assigned to it in our model's prediction.
#

# Let's review some of the most likely label errors:
#

label_issues = cl_h2o.get_label_issues()

label_issues.is_label_issue.value_counts(normalize=True)

label_issues[label_issues.is_label_issue]

# These examples appear the most suspicious to our model and should be carefully re-examined. Perhaps the original annotators missed something when deciding on the labels for these individuals.
#

# We can get predictions from the resulting model and evaluate them, just like how we did it for the original scikit-learn model.
#

acc_cl_og, auc_cl_og = eval_model(cl_h2o, X_test, y_test)
print(f"Test accuracy&auc of cleanlab's H2OAutoML: {acc_cl_og}, {auc_cl_og}")

# We can see that the test set accuracy slightly improved as a result of the data cleaning. Note that this will not always be the case, especially when we evaluate on test data that are themselves noisy. The best practice is to run `cleanlab` to identify potential label issues and then manually review them, before blindly trusting any accuracy metrics. In particular, the most effort should be made to ensure high-quality test data, which is supposed to reflect the expected performance of our model during deployment.
#

# ## Label issues in test data
#
# Use cleanlab to find label issues in test data.

te_data_health = cleanlab.dataset.health_summary(
    labels=y_test,
    pred_probs=cl_h2o.predict_proba(X_test),
    confident_joint=cl_h2o.confident_joint,
)

ranked_test_label_issues = cl_h2o.find_label_issues(
    labels=y_test,
    pred_probs=cl_h2o.predict_proba(X_test),
)

ranked_test_label_issues[ranked_test_label_issues.is_label_issue].shape

X_test_clean = np.delete(X_test, ranked_test_label_issues.is_label_issue, axis=0)
y_test_clean = np.delete(y_test, ranked_test_label_issues.is_label_issue, axis=0)
X_test.shape, X_test_clean.shape, y_test_clean.shape

acc_og_cl, auc_og_cl = eval_model(og_h2o, X_test_clean, y_test_clean)
print(
    f"Test accuracy&auc of cleanlab's H2OAutoML on cleaned test data : {acc_og_cl}, {auc_og_cl}"
)

acc_cl_cl, auc_cl_cl = eval_model(cl_h2o, X_test_clean, y_test_clean)
print(
    f"Test accuracy&auc of cleanlab's H2OAutoML on cleaned test data : {acc_cl_cl}, {auc_cl_cl}"
)

# ## Summary
#

pd.DataFrame(
    columns=["Train Data", "Test Data", "Test ACC", "Test AUC"],
    data=[
        ["Original", "Original", acc_og_og, auc_og_og],
        ["Original", "Cleanlab", acc_og_cl, auc_og_cl],
        ["Cleanlab", "Original", acc_cl_og, auc_cl_og],
        ["Cleanlab", "Cleanlab", acc_cl_cl, auc_cl_cl],
    ],
)

# + nbsphinx="hidden"
# Check that cleanlab has improved prediction performance``
assert acc_og_og < acc_cl_cl
assert auc_og_og < auc_cl_cl

# +
# h2o.cluster().shutdown()
# -

# END
