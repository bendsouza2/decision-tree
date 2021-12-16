# import DecisionTree as DecisionTree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('processed.cleveland.data', header=None)
df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal', 'hd']

# print(df.head())
#
# print(df.dtypes)  # checking data types
#
# print(df['ca'].unique)  # checking column for unique values
#
# print(df['thal'].unique) # checking column for unique values
#
# print(len(df.loc[(df['ca'] == '?')
#           |
#                  (df['thal'] == '?')]))

df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]  # importing all rows with no missing values


X = df_no_missing.drop('hd', axis=1).copy()  # data for classification
y = df_no_missing['hd'].copy()  # data to predict

X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])  # one-hot encoding


non_zero_index = y > 0  # all non-zero values of y (i.e. has heart disease)
y[non_zero_index] = 1  # setting all non-zero values equal to 1

X_train, X_test = train_test_split(X_encoded, random_state=42)
y_train, y_test = train_test_split(y, random_state=42)

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# Plotting the tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, class_names=['No HD', 'Yes HD'], feature_names=X_encoded.columns)


# Confusion matrix
# plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=['Does not have HD', 'Has HD'])

# Updated confusion matrix as plot_confusion_matrix has been deprecated
cm = ConfusionMatrixDisplay.from_estimator(clf_dt, X_test, y_test, display_labels=['Does not have HD', 'Has HD'])


# Pruning
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]  # removing maximum value

dta = []  # array to put decision trees into

# Creating a decision tree for each value of alpha
for alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    clf_dt.fit(X_train, y_train)
    dta.append(clf_dt)

# Graphing the accuracy of the trees for the training and testing dataset as a function of alpha
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in dta]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in dta]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy for different values of alpha')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()


# Cross-validation (5-fold)
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')


# Optimal alpha with cross validation
alpha_loop_vals = []  # array to store results of each fold
for alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_vals.append([alpha, np.mean(scores), np.std(scores)])

# Plot the graph
alpha_results = pd.DataFrame(alpha_loop_vals, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.show()

# Finding optimal value for alpha
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]['alpha']
ideal_ccp_alpha = float(ideal_ccp_alpha)  # converting ideal_ccp_alpha from series to float


