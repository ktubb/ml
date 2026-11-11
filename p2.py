import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
data = {
 'Age': [25, 30, np.nan, 35, 40],
 'Salary': [50000, 60000, 70000, np.nan, 80000],
 'City': ['New York', 'Paris', 'London', 'London', np.nan],
 'purchased': ['no', 'yes', 'no', 'yes', 'no']
}
df = pd.DataFrame(data)
print("1. Original Dataset (with Missing Values) : ")
print(df)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['City'] = df['City'].fillna(df['City'].mode()[0])
print("\n2. Dataset After Imputation (Mean/Mode) : ")
print(df)
df_encoded = pd.get_dummies(df, columns=['City','purchased'])
print("\n3. Dataset After One-Hot Encoding : ")
print(df_encoded)
scaler = MinMaxScaler()
df_scaled = df_encoded.copy()
df_scaled[['Age', 'Salary']] = scaler.fit_transform(df_scaled[['Age', 'Salary']])
print("\n4. Final Dataset After MinMaxScaler (0-1 Scaling) : ")
print(df_scaled)
