
# Multicollinearity of Features - Lab

## Introduction

In this lab, you'll identify multicollinearity in the Boston Housing dataset.

## Objectives
You will be able to:
* Create and Interpret a correlation matrix/heatmap and scatter matrix
* Identify if variables are exhibiting collinearity

## Correlation matrix for the Boston Housing data

Let's reimport the Boston Housing data and use the data with the categorical variables for `tax_dummy` and `rad_dummy`: 


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)

# First, create bins for RAD based on the values observed. 5 values will result in 4 bins
bins = [0, 3, 4 , 5, 24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# First, create bins for TAX based on the values observed. 6 values will result in 5 bins
bins = [0, 250, 300, 360, 460, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX", drop_first=True)
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD", drop_first=True)
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)

# Inspect the data
boston_features.head()
```

## Scatter matrix

Create the scatter matrix for the Boston Housing data: 


```python

```

This took a while to load. Not surprisingly, the categorical variables didn't really provide any meaningful result. Remove the categorical columns associated with `'RAD'` and `'TAX'` from the data again and look at the scatter matrix again: 


```python

```


```python

```

## Correlation matrix

Next, let's look at the correlation matrix: 


```python

```

Return `True` for positive or negative correlations that are bigger than 0.75: 


```python

```

Remove the most problematic feature from the data: 


```python

```

## Summary
Good job! You got some hands-on practice creating and interpreting a scatter matrix and correlation matrix to identify if variables are collinear in the Boston Housing data set. You also edited the Boston Housing data set so highly correlated variables are removed.
