import pandas as pd
from sklearn.model_selection import train_test_split

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

df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')] # importing all rows with no missing values


X = df_no_missing.drop('hd', axis=1).copy()  # data for classification
y = df_no_missing['hd'].copy()  # data to predict

X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])  # one-hot encoding


non_zero_index = y > 0  # all non-zero values of y (i.e. has heart disease)
y[non_zero_index] = 1  # setting all non-zero values equal to 1

X_train, X_test = train_test_split(X_encoded, random_state=42)
y_train, y_test = train_test_split(y, random_state=42)
