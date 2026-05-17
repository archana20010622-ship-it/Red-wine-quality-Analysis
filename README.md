# 🍷 Red Wine Quality Analysis using Python

## 📌 Project Overview
This project performs **Exploratory Data Analysis (EDA)** on the Red Wine Quality dataset using Python.  
The analysis includes:

- Data Cleaning
- Correlation Analysis
- Outlier Detection using IQR Method
- Data Visualization
- Class Imbalance Detection
- Balancing Dataset using SMOTE

---

# 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 📂 Dataset Loading

```python
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv('winequality-red.csv')
```

---

# 🔍 Basic Data Analysis

## First 5 Rows

```python
print(df.head(5))
```

## Dataset Summary

```python
print(df.info())
```

## Statistical Summary

```python
print(df.describe())
```

## Shape of Dataset

```python
print(f"Shape of dataframe: {df.shape}")
```

## Column Names

```python
print(df.columns)
```

## Unique Values in Quality Column

```python
print(df['quality'].unique())
```

---

# ❌ Missing Values Analysis

```python
print(df.isnull().sum())
```

### Observation
No missing values were found in the dataset.

---

# 🔁 Duplicate Records

## Finding Duplicate Rows

```python
print(df[df.duplicated()])
```

## Removing Duplicate Rows

```python
df.drop_duplicates(inplace=True)
```

---

# 📊 Correlation Analysis

## Full Correlation Matrix

```python
print(df.corr())
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
plt.show()
```

### 📷 Correlation Matrix with All Features

![Correlation Matrix](https://github.com/user-attachments/assets/8edaa973-1d91-461a-8547-76fc6965e5e0)



## Correlation with Quality Feature

```python
print(df.corr()['quality'])
plt.figure(figsize=(10,6))
sns.heatmap(df.corr()[['quality']], annot=True)
plt.show()
```

### 📷 Correlation with Quality Feature

![Correlation Heatmap](https://github.com/user-attachments/assets/132283b6-50ae-4125-8532-8855c6af8faf)


### 📌 Key Findings
- Alcohol has the strongest positive correlation with wine quality.
- Volatile acidity has a strong negative correlation with quality.
- Sulphates and citric acid moderately improve wine quality.
- Strong inter-feature correlations exist between fixed acidity and pH.

---
# 🚨 Outlier Detection using IQR Method

## Detecting Outliers in Residual Sugar Feature

```python
minimum, Q1, median, Q3, maximum = np.quantile(
    df['residual sugar'],
    [0, 0.25, 0.5, 0.75, 1]
)

IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

outliers = df[
    (df['residual sugar'] < lower_fence) |
    (df['residual sugar'] > upper_fence)
]

print(outliers[['residual sugar']])
```

# 📦 Boxplot Visualization

```python
sns.boxplot(df['residual sugar'])
plt.show()
```

### 📷 Residual Sugar Outliers

![Outlier Boxplot](https://github.com/user-attachments/assets/33c744a5-9a5b-42ed-b896-6bff2a4bfb99)

### 📌 Outlier Analysis Conclusion
- Residual sugar contains several upper outliers.
- The feature shows a right-skewed distribution.
- Some wines contain unusually high sugar content compared to the majority.

---

# ⚖️ Class Imbalance Analysis

```python
print(df['quality'].value_counts().sort_index())
```

### 📌 Observation
The Red Wine dataset is imbalanced because some quality categories contain significantly more samples than others.

# 🤖 Handling Imbalance using SMOTE

```python
x = df.drop('quality', axis=1)
y = df['quality']

oversample = SMOTE()

x_sampled, y_sampled = oversample.fit_resample(x, y)

df2 = pd.concat([x_sampled, y_sampled], axis=1)

print(df2['quality'].value_counts().sort_index())
```

### 📌 SMOTE Conclusion
- Applied SMOTE to balance minority quality classes.
- Improved dataset balance for machine learning classification tasks.

---

# 🧰 Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SMOTE

---

# ✅ Final Conclusion

This project demonstrates practical **Exploratory Data Analysis (EDA)** techniques including:

- Correlation Analysis
- Heatmap Visualization for Correlation Analysis
- Outlier Detection using IQR
- Boxplot Visualization for Outlier Detection
- Class Imbalance Analysis
- Dataset Balancing using SMOTE

The analysis identified alcohol, volatile acidity, sulphates, and citric acid as important features influencing wine quality.
