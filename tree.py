import pandas as pd

df = pd.read_csv('processed.cleveland.data', header=None)
print(df.head())

df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', ' slope', 'ca',
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
print(df_no_missing.head)