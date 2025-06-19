# 📦 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Step 1: Load the Dataset
df = pd.read_csv("SalesDataset_WithOutliers.csv")
df.head()

# 📊 Step 2: IQR Method for Outlier Detection in 'TotalSale'
Q1 = df['TotalSale'].quantile(0.25)
Q3 = df['TotalSale'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 🚩 Mark outliers
df['Outlier'] = ((df['TotalSale'] < lower_bound) | (df['TotalSale'] > upper_bound)).astype(int)

# 📋 Summary of outliers
print(df['Outlier'].value_counts())

# 📈 Step 3: Visualize Outliers using Boxplot
sns.boxplot(x=df['TotalSale'])
plt.title("Boxplot of TotalSale (Outlier Detection)")
plt.xlabel("TotalSale Amount")
plt.grid(True)
plt.show()

# 💾 Step 4: Save the Updated Dataset (Optional)
df.to_csv("SalesDataset_WithOutliers.csv", index=False)

# 🔍 View Outlier Records
print(df[df["Outlier"] == 1])
