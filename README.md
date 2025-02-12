# Data Preprocessing for Machine Learning

This project demonstrates essential data preprocessing steps for preparing a dataset before training a machine learning model. The steps include handling missing data, encoding categorical variables, splitting data into training and test sets, and feature scaling.

## Libraries Used
```python
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For handling datasets in DataFrame format
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler  # For encoding and scaling
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
```

## Steps in Data Preprocessing

### 1. Importing the Dataset
```python
datasets = pd.read_csv('Data.csv')  # Load dataset into a DataFrame
x = datasets.iloc[:, :-1].values  # Extract independent variables (all columns except the last one)
y = datasets.iloc[:, -1].values  # Extract dependent variable (last column)
```

### 2. Handling Missing Data
We use `SimpleImputer` to replace missing values with different strategies.
```python
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_mode = SimpleImputer(strategy='most_frequent')
imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
```
Applying mean imputation on numerical columns (e.g., age, salary):
```python
imputer_mean.fit(x[:, 1:3])
x[:, 1:3] = imputer_mean.transform(x[:, 1:3])
```

### 3. Encoding Categorical Data
Since machine learning models work with numerical data, we encode categorical columns using One-Hot Encoding and Label Encoding.

#### One-Hot Encoding for Independent Variables
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
```
This transforms categorical features into a binary matrix representation.

#### Dummy Variable Trap
One-hot encoding can introduce redundancy by creating highly correlated variables. The Dummy Variable Trap occurs when one category can be predicted from the others. To avoid this, we typically drop one dummy variable column manually:
```python
x = x[:, 1:]
```

#### Label Encoding for Dependent Variable
```python
le = LabelEncoder()
y = le.fit_transform(y)
```
This converts categorical output labels into numerical values.

### 4. Splitting Dataset into Training and Testing Sets
We split the dataset into training and test sets in an 80:20 ratio to evaluate the model's performance.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```
- `x_train`, `y_train`: Used for training the model
- `x_test`, `y_test`: Used for evaluating the model

### 5. Feature Scaling
Feature scaling ensures all numerical features have comparable magnitudes to improve model performance.
```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```

#### Difference Between `fit_transform` and `transform`
1. `fit_transform(X_train)`
   - Computes the scaling parameters (mean and standard deviation for StandardScaler) from `X_train` and applies the transformation.
   - Used on the training set to standardize data.

2. `transform(X_test)`
   - Applies the previously computed parameters from `X_train` to `X_test`.
   - Ensures that the test data is scaled using the same parameters as the training data to maintain consistency and prevent data leakage.

#### Why Feature Scaling?
- It standardizes data to have a mean of 0 and a standard deviation of 1.
- It improves model convergence for algorithms that rely on distance calculations (e.g., KNN, SVM, Logistic Regression).
- It prevents features with large values from dominating others with smaller ranges.

#### Why Feature Scaling is Done After Splitting the Dataset?
Feature scaling is applied after splitting to prevent data leakage. If scaling were done before splitting, the test set would be influenced by information from the training set, leading to biased model performance. By scaling separately, we ensure a fair and independent evaluation of the model.

## Summary
This repository provides a structured pipeline for preprocessing data before feeding it into a machine learning model. The steps ensure that the dataset is clean, well-encoded, and scaled for optimal model performance.

![alt text](https://github.com/tanveerj5/Data-Preprocessing/blob/main/Data%20Preprocessing%20Setps.png)

