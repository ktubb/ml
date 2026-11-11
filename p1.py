import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
iris_dataset = load_iris()
print("Data shape (NumPy):", iris_dataset.data.shape)
print("Mean Sepal Length: {:.2f} cm".format(np.mean(iris_dataset.data[:, 0])))
iris_df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris_dataset.target, iris_dataset.target_names)
print("\nFirst five rows:")
print(iris_df.head())
print("\nStatistical summary:")
print(iris_df.describe())
sns.pairplot(iris_df, hue='species', palette='Set2', markers=['o', 's', 'D'])
plt.suptitle("Iris Feature Relationships (Seaborn Pairplot)", y=1.02)
plt.show()
