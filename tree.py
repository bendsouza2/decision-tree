import DecisionTree as DecisionTree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
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
plt.show()

# Confusion matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=['Does not have HD', 'Has HD'])


