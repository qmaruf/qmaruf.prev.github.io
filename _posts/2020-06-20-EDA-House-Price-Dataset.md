---
title: House_prices_advanced_regression_techniques
date: 2020-06-20 00:00:00 Z
---

![Advanced regression techniques](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
In this EDA I'll explore a house price dataset. The dataset is collected from a Kaggle playground competition. You can check the dataset from here. https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data. I'll try to analyse some variables and their impact to determine the house price. Besides, I'll try to do some basic data cleaning.

First thing first. Let's import the necessary libraries.




```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats
import xgboost
%matplotlib inline
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm



```
# load the data
train_df = pd.read_csv('./drive/My Drive/kaggle/train.csv')
test_df = pd.read_csv('./drive/My Drive/kaggle/test.csv')
```


```
print ('%d rows and %d columns in training data'%(train_df.shape[0], train_df.shape[1]))
print ('%d rows and %d columns in testing data'%(test_df.shape[0], test_df.shape[1]))
```

    1460 rows and 81 columns in training data
    1459 rows and 80 columns in testing data



```
# columns in the data
print (train_df.columns)
```

    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')


Id column is unnecessary for our cause. Remove this column first.


```
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)
```

Our goal is to predict the *SalePrice* using the other features. Let's look at the *SalePrice* column.


```
train_df.SalePrice.describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



Well, the minimum value in the *SalePrice* is 34900. That means there is no negative or zero value in the target. We are good to go. We can look at the distribution now.


```
sns.distplot(train_df.SalePrice)
plt.axvline(train_df.SalePrice.mean(), linestyle='--', c='g', label='mean SalePrice')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f0317513a90>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_10_1.png)


Ok, the *SalePrice* is not normally distributed and shows [positive skewness](https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa).

Let's look at a snapshot of the training data.


```
train_df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1073</th>
      <td>60</td>
      <td>RL</td>
      <td>75.0</td>
      <td>7950</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>6</td>
      <td>1977</td>
      <td>1977</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>Plywood</td>
      <td>BrkFace</td>
      <td>140.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>BLQ</td>
      <td>535</td>
      <td>Unf</td>
      <td>0</td>
      <td>155</td>
      <td>690</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>698</td>
      <td>728</td>
      <td>0</td>
      <td>1426</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1977.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>440</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>252</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>159500</td>
    </tr>
    <tr>
      <th>953</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>11075</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>4</td>
      <td>1969</td>
      <td>1969</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>232.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>Av</td>
      <td>ALQ</td>
      <td>562</td>
      <td>LwQ</td>
      <td>193</td>
      <td>29</td>
      <td>784</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1168</td>
      <td>800</td>
      <td>0</td>
      <td>1968</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min2</td>
      <td>1</td>
      <td>Po</td>
      <td>Attchd</td>
      <td>1969.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>530</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>305</td>
      <td>189</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>400</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>352</th>
      <td>50</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9084</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>6</td>
      <td>1941</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>Fa</td>
      <td>Mn</td>
      <td>LwQ</td>
      <td>236</td>
      <td>Rec</td>
      <td>380</td>
      <td>0</td>
      <td>616</td>
      <td>GasA</td>
      <td>TA</td>
      <td>N</td>
      <td>SBrkr</td>
      <td>616</td>
      <td>495</td>
      <td>0</td>
      <td>1111</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1941.0</td>
      <td>Unf</td>
      <td>1</td>
      <td>200</td>
      <td>TA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2008</td>
      <td>ConLw</td>
      <td>Normal</td>
      <td>95000</td>
    </tr>
    <tr>
      <th>196</th>
      <td>20</td>
      <td>RL</td>
      <td>79.0</td>
      <td>9416</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>205.0</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>1126</td>
      <td>Unf</td>
      <td>0</td>
      <td>600</td>
      <td>1726</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1726</td>
      <td>0</td>
      <td>0</td>
      <td>1726</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Ex</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2007.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>786</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>171</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>266</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>311872</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20</td>
      <td>RL</td>
      <td>91.0</td>
      <td>10652</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>Stone</td>
      <td>306.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1494</td>
      <td>1494</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1494</td>
      <td>0</td>
      <td>0</td>
      <td>1494</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2006.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>840</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>160</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>279500</td>
    </tr>
    <tr>
      <th>653</th>
      <td>50</td>
      <td>RM</td>
      <td>60.0</td>
      <td>10320</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>IDOTRR</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>6</td>
      <td>7</td>
      <td>1906</td>
      <td>1995</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>756</td>
      <td>756</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>756</td>
      <td>713</td>
      <td>0</td>
      <td>1469</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1906.0</td>
      <td>Unf</td>
      <td>1</td>
      <td>216</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>57</td>
      <td>0</td>
      <td>239</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>135000</td>
    </tr>
    <tr>
      <th>355</th>
      <td>20</td>
      <td>RL</td>
      <td>105.0</td>
      <td>11249</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1995</td>
      <td>1995</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>334</td>
      <td>BLQ</td>
      <td>544</td>
      <td>322</td>
      <td>1200</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1200</td>
      <td>0</td>
      <td>0</td>
      <td>1200</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1995.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>521</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>177500</td>
    </tr>
    <tr>
      <th>882</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>9636</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>1992</td>
      <td>1993</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>808</td>
      <td>808</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>808</td>
      <td>785</td>
      <td>0</td>
      <td>1593</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>BuiltIn</td>
      <td>1993.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>389</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>342</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>178000</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>20</td>
      <td>FV</td>
      <td>72.0</td>
      <td>8640</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>8</td>
      <td>5</td>
      <td>2009</td>
      <td>2009</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>72.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>936</td>
      <td>Unf</td>
      <td>0</td>
      <td>364</td>
      <td>1300</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1314</td>
      <td>0</td>
      <td>0</td>
      <td>1314</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2009.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>552</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>135</td>
      <td>112</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2009</td>
      <td>New</td>
      <td>Partial</td>
      <td>229456</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>40</td>
      <td>RL</td>
      <td>NaN</td>
      <td>23595</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Sev</td>
      <td>ClearCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>6</td>
      <td>1979</td>
      <td>1979</td>
      <td>Shed</td>
      <td>WdShake</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1258</td>
      <td>Unf</td>
      <td>0</td>
      <td>74</td>
      <td>1332</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1332</td>
      <td>192</td>
      <td>0</td>
      <td>1524</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Gd</td>
      <td>4</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1979.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>586</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>268</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>260000</td>
    </tr>
  </tbody>
</table>
</div>



There are both numerical and categorical features in the dataset. Now we will group all the features into these two groups. All *object* type columns will be considered as categorical, rest of the columns are numerical.


```
train_df_categorical = train_df.select_dtypes(include='object')
train_df_numerical = train_df.select_dtypes(exclude='object')
```


```
train_df_categorical
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>KitchenQual</th>
      <th>Functional</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NWAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>Stone</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>Rec</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Min1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>Ex</td>
      <td>Gd</td>
      <td>Stone</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>Rec</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>None</td>
      <td>Gd</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>BLQ</td>
      <td>LwQ</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>Fin</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 43 columns</p>
</div>




```
train_df_numerical
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2003.0</td>
      <td>2</td>
      <td>548</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1976.0</td>
      <td>2</td>
      <td>460</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2001.0</td>
      <td>2</td>
      <td>608</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1998.0</td>
      <td>3</td>
      <td>642</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>2000.0</td>
      <td>3</td>
      <td>836</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>60</td>
      <td>62.0</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>953</td>
      <td>953</td>
      <td>953</td>
      <td>694</td>
      <td>0</td>
      <td>1647</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1999.0</td>
      <td>2</td>
      <td>460</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>20</td>
      <td>85.0</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>119.0</td>
      <td>790</td>
      <td>163</td>
      <td>589</td>
      <td>1542</td>
      <td>2073</td>
      <td>0</td>
      <td>0</td>
      <td>2073</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1978.0</td>
      <td>2</td>
      <td>500</td>
      <td>349</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>70</td>
      <td>66.0</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>0.0</td>
      <td>275</td>
      <td>0</td>
      <td>877</td>
      <td>1152</td>
      <td>1188</td>
      <td>1152</td>
      <td>0</td>
      <td>2340</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>1941.0</td>
      <td>1</td>
      <td>252</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>20</td>
      <td>68.0</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>0.0</td>
      <td>49</td>
      <td>1029</td>
      <td>0</td>
      <td>1078</td>
      <td>1078</td>
      <td>0</td>
      <td>0</td>
      <td>1078</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1950.0</td>
      <td>1</td>
      <td>240</td>
      <td>366</td>
      <td>0</td>
      <td>112</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>20</td>
      <td>75.0</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>0.0</td>
      <td>830</td>
      <td>290</td>
      <td>136</td>
      <td>1256</td>
      <td>1256</td>
      <td>0</td>
      <td>0</td>
      <td>1256</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1965.0</td>
      <td>1</td>
      <td>276</td>
      <td>736</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 37 columns</p>
</div>



The next question is how the numerical columns are related to the *SalePrice* column? We can check this relationship using a correlation matrix. Only showing the correlation larger than 0.5


```
corr = train_df_numerical.corr()
corr = corr[corr > 0.5]
plt.figure(figsize=(20, 14))
sns.heatmap(corr, annot=True, fmt='.2f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0319e62e80>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_19_1.png)


Now we will check a smaller version of the above correlation matrix. We will visualize the correlation among the top 10 highly correation fetatures with the *SalePrice*.


```
k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df_numerical[cols].values.T)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```


![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_21_0.png)


Well, *OverallQual* and *GrLivArea* are two most strongly correlated features with *SalePrice*.

*GarageArea* and *GarageCars* are highly correalted. It seems number of cars in the garage is related to the garage size.

*TotalBsmtSF* and *1stFlrSF* are strongly correlated too. Seems normal.

*FullBath* and *TotRmsAbvGrd* are highly correlated to *GrLivArea*. This relation is expected too.

IMO, *YearBuilt* should have a higher correlation with *SalePrice*. But it seems pretty low here. We will investigate this property late.




```

```

Now we will draw some plots to visualize the relationship between *SalePrice* and some highly correlated features.


```
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df_numerical[cols], height = 1.5) 
```




    <seaborn.axisgrid.PairGrid at 0x7f0315b53400>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_25_1.png)


Now we will perform some missing data analysis. Let's find out how many data are missing in each feature.


```
total = train_df.isnull().sum().sort_values(ascending=False)
missing_df = pd.DataFrame(total).reset_index()
missing_df.columns = ['feature', 'missing values']
missing_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>missing values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PoolQC</td>
      <td>1453</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MiscFeature</td>
      <td>1406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alley</td>
      <td>1369</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fence</td>
      <td>1179</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FireplaceQu</td>
      <td>690</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LotFrontage</td>
      <td>259</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GarageType</td>
      <td>81</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GarageCond</td>
      <td>81</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GarageFinish</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GarageQual</td>
      <td>81</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GarageYrBlt</td>
      <td>81</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BsmtFinType2</td>
      <td>38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BsmtExposure</td>
      <td>38</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BsmtQual</td>
      <td>37</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BsmtCond</td>
      <td>37</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BsmtFinType1</td>
      <td>37</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MasVnrArea</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MasVnrType</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Electrical</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RoofMatl</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



*GarageType*, *GarageCond*, *GarageFinish*, *GarageQual* and *GarageYrBlt* has same amount of missing values! Seems like all of these are derived from a similar source.

Besides, *BsmtQual*, *BsmtCond* and *BsmtFinType1* has similar number of missing values too.

Just learnt about the *info* function. It shows a concise summary of a dataframe. Check this out. It shows the data type and number of non-null values for each features.


```
print (train_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1460 non-null   int64  
     1   MSZoning       1460 non-null   object 
     2   LotFrontage    1201 non-null   float64
     3   LotArea        1460 non-null   int64  
     4   Street         1460 non-null   object 
     5   Alley          91 non-null     object 
     6   LotShape       1460 non-null   object 
     7   LandContour    1460 non-null   object 
     8   Utilities      1460 non-null   object 
     9   LotConfig      1460 non-null   object 
     10  LandSlope      1460 non-null   object 
     11  Neighborhood   1460 non-null   object 
     12  Condition1     1460 non-null   object 
     13  Condition2     1460 non-null   object 
     14  BldgType       1460 non-null   object 
     15  HouseStyle     1460 non-null   object 
     16  OverallQual    1460 non-null   int64  
     17  OverallCond    1460 non-null   int64  
     18  YearBuilt      1460 non-null   int64  
     19  YearRemodAdd   1460 non-null   int64  
     20  RoofStyle      1460 non-null   object 
     21  RoofMatl       1460 non-null   object 
     22  Exterior1st    1460 non-null   object 
     23  Exterior2nd    1460 non-null   object 
     24  MasVnrType     1452 non-null   object 
     25  MasVnrArea     1452 non-null   float64
     26  ExterQual      1460 non-null   object 
     27  ExterCond      1460 non-null   object 
     28  Foundation     1460 non-null   object 
     29  BsmtQual       1423 non-null   object 
     30  BsmtCond       1423 non-null   object 
     31  BsmtExposure   1422 non-null   object 
     32  BsmtFinType1   1423 non-null   object 
     33  BsmtFinSF1     1460 non-null   int64  
     34  BsmtFinType2   1422 non-null   object 
     35  BsmtFinSF2     1460 non-null   int64  
     36  BsmtUnfSF      1460 non-null   int64  
     37  TotalBsmtSF    1460 non-null   int64  
     38  Heating        1460 non-null   object 
     39  HeatingQC      1460 non-null   object 
     40  CentralAir     1460 non-null   object 
     41  Electrical     1459 non-null   object 
     42  1stFlrSF       1460 non-null   int64  
     43  2ndFlrSF       1460 non-null   int64  
     44  LowQualFinSF   1460 non-null   int64  
     45  GrLivArea      1460 non-null   int64  
     46  BsmtFullBath   1460 non-null   int64  
     47  BsmtHalfBath   1460 non-null   int64  
     48  FullBath       1460 non-null   int64  
     49  HalfBath       1460 non-null   int64  
     50  BedroomAbvGr   1460 non-null   int64  
     51  KitchenAbvGr   1460 non-null   int64  
     52  KitchenQual    1460 non-null   object 
     53  TotRmsAbvGrd   1460 non-null   int64  
     54  Functional     1460 non-null   object 
     55  Fireplaces     1460 non-null   int64  
     56  FireplaceQu    770 non-null    object 
     57  GarageType     1379 non-null   object 
     58  GarageYrBlt    1379 non-null   float64
     59  GarageFinish   1379 non-null   object 
     60  GarageCars     1460 non-null   int64  
     61  GarageArea     1460 non-null   int64  
     62  GarageQual     1379 non-null   object 
     63  GarageCond     1379 non-null   object 
     64  PavedDrive     1460 non-null   object 
     65  WoodDeckSF     1460 non-null   int64  
     66  OpenPorchSF    1460 non-null   int64  
     67  EnclosedPorch  1460 non-null   int64  
     68  3SsnPorch      1460 non-null   int64  
     69  ScreenPorch    1460 non-null   int64  
     70  PoolArea       1460 non-null   int64  
     71  PoolQC         7 non-null      object 
     72  Fence          281 non-null    object 
     73  MiscFeature    54 non-null     object 
     74  MiscVal        1460 non-null   int64  
     75  MoSold         1460 non-null   int64  
     76  YrSold         1460 non-null   int64  
     77  SaleType       1460 non-null   object 
     78  SaleCondition  1460 non-null   object 
     79  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(34), object(43)
    memory usage: 912.6+ KB
    None


We want to see a boxplot of our *SalePrice* column. Here it is. Its an enhanced version of boxplot. The left and the rightmost points are the outliers.


```
sns.boxenplot(train_df.SalePrice)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f031333a198>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_32_1.png)


Let's analyze the *OverallQual* feature. First thing first. What is the distribution of this feature?


```
sns.distplot(train_df.OverallQual)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f03132b7b70>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_34_1.png)


Rating *5* is the most common in the *OverallQual* feature. Let's draw a boxplot to better visualize the relationship between *OverallQual* and *SalePrice*. For each quality it will show the five number summary (minimum, maximum, median, first and third quartile) of corresponding *SalePrice*. We can have a sense of data dispersion and outlier from this plot.


```
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0313261dd8>




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_36_1.png)


Now we'll plot the correlation of *SalePrice* with all other features.


```
corr = train_df.corr()
SP_corr = corr['SalePrice'].sort_values(ascending=False)[1:]

plt.figure(figsize=(10, 15))
SP_corr.plot(kind='barh')
plt.title('correlation with SalePrice')
```




    Text(0.5, 1.0, 'correlation with SalePrice')




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_38_1.png)


We've seen that *SalePrice* is not normally distributed. We'll perform log transform on this target to see the impact.


```
saleprice_lt = np.log(train_df['SalePrice'])

fig ,axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axs = np.array(axs)
sns.distplot(saleprice_lt, ax=axs[1])
sns.distplot(train_df['SalePrice'], ax=axs[0])
axs[0].set_title('SalePrice')
axs[1].set_title('Log transformed SalePrice')
```




    Text(0.5, 1.0, 'Log transformed SalePrice')




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_40_1.png)


We'll use the log transformed *SalePrice* for modelling purpose.

Now, we'll use xgboost regression on this dataset to find the validation accuracy and importance of each features for predicting the house price. This is a very basic model without any feature engineering.


```
y = train_df.SalePrice
X = train_df.drop('SalePrice', axis=1)
```

Before starting the modelling, we need to find the categorical features and change the to one hot encoded features. Here is our categorical and numerical features.


```
train_df_categorical = train_df.select_dtypes(include='object')
train_df_numerical = train_df.select_dtypes(exclude='object')
```

Let's check how many unique values in each of the features.


```
cat_unique = train_df_categorical.nunique()
plt.figure(figsize=(20, 5))
sns.barplot(x=cat_unique.index, y=cat_unique.values)
plt.xticks(rotation=45)
plt.title('number of unique values in categorical features')
```




    Text(0.5, 1.0, 'number of unique values in categorical features')




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_47_1.png)



```
num_unique = train_df_numerical.nunique()
plt.figure(figsize=(20, 5))
sns.barplot(x=num_unique.index, y=num_unique.values)
plt.xticks(rotation=45)
plt.title('number of unique values in numerical features')
```




    Text(0.5, 1.0, 'number of unique values in numerical features')




![png](Kaggle___House_Prices_Advanced_Regression_Techniques_files/Kaggle___House_Prices_Advanced_Regression_Techniques_48_1.png)


Well, in some of the numerical column the number of unique features is very low. May be they are categorical features in numerical format.

To keep things simpler, we will use pandas to_dummies() function to convert the whole dataset into ohe format. Before that, we should replace the missing values.


```
train_df = train_df.fillna(-999)
```


```
y = train_df.SalePrice
X = train_df.iloc[:, :-1]
X = pd.get_dummies(X)
```


```
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
```


```
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
```


```
model = XGBRegressor(
    max_depth=6, 
    n_estimators=512, 
    random_state=1,colsample_bytree=0.8, learning_rate=0.001
    )
```


```
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          early_stopping_rounds=15)
```

    [16:27:36] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:199943
    Will train until validation_0-rmse hasn't improved in 15 rounds.
    [1]	validation_0-rmse:199762
    [2]	validation_0-rmse:199579
    [3]	validation_0-rmse:199398
    [4]	validation_0-rmse:199214
    [5]	validation_0-rmse:199033
    [6]	validation_0-rmse:198853
    [7]	validation_0-rmse:198675
    [8]	validation_0-rmse:198495
    [9]	validation_0-rmse:198314
    [10]	validation_0-rmse:198133
    [11]	validation_0-rmse:197952
    [12]	validation_0-rmse:197772
    [13]	validation_0-rmse:197593
    [14]	validation_0-rmse:197413
    [15]	validation_0-rmse:197236
    [16]	validation_0-rmse:197056
    [17]	validation_0-rmse:196877
    [18]	validation_0-rmse:196698
    [19]	validation_0-rmse:196519
    [20]	validation_0-rmse:196344
    [21]	validation_0-rmse:196168
    [22]	validation_0-rmse:195989
    [23]	validation_0-rmse:195810
    [24]	validation_0-rmse:195633
    [25]	validation_0-rmse:195458
    [26]	validation_0-rmse:195281
    [27]	validation_0-rmse:195107
    [28]	validation_0-rmse:194929
    [29]	validation_0-rmse:194752
    [30]	validation_0-rmse:194573
    [31]	validation_0-rmse:194400
    [32]	validation_0-rmse:194224
    [33]	validation_0-rmse:194044
    [34]	validation_0-rmse:193867
    [35]	validation_0-rmse:193692
    [36]	validation_0-rmse:193515
    [37]	validation_0-rmse:193340
    [38]	validation_0-rmse:193165
    [39]	validation_0-rmse:192992
    [40]	validation_0-rmse:192817
    [41]	validation_0-rmse:192643
    [42]	validation_0-rmse:192467
    [43]	validation_0-rmse:192293
    [44]	validation_0-rmse:192117
    [45]	validation_0-rmse:191946
    [46]	validation_0-rmse:191773
    [47]	validation_0-rmse:191599
    [48]	validation_0-rmse:191425
    [49]	validation_0-rmse:191252
    [50]	validation_0-rmse:191082
    [51]	validation_0-rmse:190912
    [52]	validation_0-rmse:190739
    [53]	validation_0-rmse:190566
    [54]	validation_0-rmse:190395
    [55]	validation_0-rmse:190223
    [56]	validation_0-rmse:190052
    [57]	validation_0-rmse:189881
    [58]	validation_0-rmse:189707
    [59]	validation_0-rmse:189534
    [60]	validation_0-rmse:189364
    [61]	validation_0-rmse:189192
    [62]	validation_0-rmse:189021
    [63]	validation_0-rmse:188851
    [64]	validation_0-rmse:188684
    [65]	validation_0-rmse:188511
    [66]	validation_0-rmse:188342
    [67]	validation_0-rmse:188171
    [68]	validation_0-rmse:188003
    [69]	validation_0-rmse:187836
    [70]	validation_0-rmse:187670
    [71]	validation_0-rmse:187501
    [72]	validation_0-rmse:187334
    [73]	validation_0-rmse:187167
    [74]	validation_0-rmse:186998
    [75]	validation_0-rmse:186828
    [76]	validation_0-rmse:186659
    [77]	validation_0-rmse:186490
    [78]	validation_0-rmse:186320
    [79]	validation_0-rmse:186151
    [80]	validation_0-rmse:185984
    [81]	validation_0-rmse:185820
    [82]	validation_0-rmse:185654
    [83]	validation_0-rmse:185488
    [84]	validation_0-rmse:185320
    [85]	validation_0-rmse:185154
    [86]	validation_0-rmse:184990
    [87]	validation_0-rmse:184823
    [88]	validation_0-rmse:184658
    [89]	validation_0-rmse:184491
    [90]	validation_0-rmse:184324
    [91]	validation_0-rmse:184160
    [92]	validation_0-rmse:183994
    [93]	validation_0-rmse:183827
    [94]	validation_0-rmse:183659
    [95]	validation_0-rmse:183493
    [96]	validation_0-rmse:183328
    [97]	validation_0-rmse:183165
    [98]	validation_0-rmse:183001
    [99]	validation_0-rmse:182834
    [100]	validation_0-rmse:182671
    [101]	validation_0-rmse:182508
    [102]	validation_0-rmse:182344
    [103]	validation_0-rmse:182179
    [104]	validation_0-rmse:182017
    [105]	validation_0-rmse:181857
    [106]	validation_0-rmse:181692
    [107]	validation_0-rmse:181530
    [108]	validation_0-rmse:181369
    [109]	validation_0-rmse:181205
    [110]	validation_0-rmse:181040
    [111]	validation_0-rmse:180879
    [112]	validation_0-rmse:180715
    [113]	validation_0-rmse:180554
    [114]	validation_0-rmse:180392
    [115]	validation_0-rmse:180232
    [116]	validation_0-rmse:180070
    [117]	validation_0-rmse:179907
    [118]	validation_0-rmse:179744
    [119]	validation_0-rmse:179580
    [120]	validation_0-rmse:179418
    [121]	validation_0-rmse:179258
    [122]	validation_0-rmse:179094
    [123]	validation_0-rmse:178931
    [124]	validation_0-rmse:178767
    [125]	validation_0-rmse:178607
    [126]	validation_0-rmse:178447
    [127]	validation_0-rmse:178284
    [128]	validation_0-rmse:178120
    [129]	validation_0-rmse:177959
    [130]	validation_0-rmse:177798
    [131]	validation_0-rmse:177644
    [132]	validation_0-rmse:177484
    [133]	validation_0-rmse:177322
    [134]	validation_0-rmse:177162
    [135]	validation_0-rmse:177003
    [136]	validation_0-rmse:176843
    [137]	validation_0-rmse:176682
    [138]	validation_0-rmse:176525
    [139]	validation_0-rmse:176367
    [140]	validation_0-rmse:176208
    [141]	validation_0-rmse:176049
    [142]	validation_0-rmse:175891
    [143]	validation_0-rmse:175729
    [144]	validation_0-rmse:175574
    [145]	validation_0-rmse:175418
    [146]	validation_0-rmse:175260
    [147]	validation_0-rmse:175103
    [148]	validation_0-rmse:174947
    [149]	validation_0-rmse:174791
    [150]	validation_0-rmse:174634
    [151]	validation_0-rmse:174475
    [152]	validation_0-rmse:174320
    [153]	validation_0-rmse:174164
    [154]	validation_0-rmse:174007
    [155]	validation_0-rmse:173850
    [156]	validation_0-rmse:173696
    [157]	validation_0-rmse:173540
    [158]	validation_0-rmse:173384
    [159]	validation_0-rmse:173230
    [160]	validation_0-rmse:173072
    [161]	validation_0-rmse:172916
    [162]	validation_0-rmse:172761
    [163]	validation_0-rmse:172603
    [164]	validation_0-rmse:172447
    [165]	validation_0-rmse:172293
    [166]	validation_0-rmse:172138
    [167]	validation_0-rmse:171984
    [168]	validation_0-rmse:171832
    [169]	validation_0-rmse:171678
    [170]	validation_0-rmse:171528
    [171]	validation_0-rmse:171374
    [172]	validation_0-rmse:171220
    [173]	validation_0-rmse:171066
    [174]	validation_0-rmse:170912
    [175]	validation_0-rmse:170759
    [176]	validation_0-rmse:170606
    [177]	validation_0-rmse:170452
    [178]	validation_0-rmse:170301
    [179]	validation_0-rmse:170147
    [180]	validation_0-rmse:169995
    [181]	validation_0-rmse:169842
    [182]	validation_0-rmse:169691
    [183]	validation_0-rmse:169538
    [184]	validation_0-rmse:169386
    [185]	validation_0-rmse:169234
    [186]	validation_0-rmse:169083
    [187]	validation_0-rmse:168934
    [188]	validation_0-rmse:168787
    [189]	validation_0-rmse:168636
    [190]	validation_0-rmse:168485
    [191]	validation_0-rmse:168335
    [192]	validation_0-rmse:168189
    [193]	validation_0-rmse:168040
    [194]	validation_0-rmse:167891
    [195]	validation_0-rmse:167744
    [196]	validation_0-rmse:167592
    [197]	validation_0-rmse:167446
    [198]	validation_0-rmse:167298
    [199]	validation_0-rmse:167147
    [200]	validation_0-rmse:166996
    [201]	validation_0-rmse:166847
    [202]	validation_0-rmse:166699
    [203]	validation_0-rmse:166550
    [204]	validation_0-rmse:166402
    [205]	validation_0-rmse:166254
    [206]	validation_0-rmse:166104
    [207]	validation_0-rmse:165957
    [208]	validation_0-rmse:165810
    [209]	validation_0-rmse:165662
    [210]	validation_0-rmse:165515
    [211]	validation_0-rmse:165367
    [212]	validation_0-rmse:165223
    [213]	validation_0-rmse:165076
    [214]	validation_0-rmse:164930
    [215]	validation_0-rmse:164788
    [216]	validation_0-rmse:164640
    [217]	validation_0-rmse:164494
    [218]	validation_0-rmse:164349
    [219]	validation_0-rmse:164201
    [220]	validation_0-rmse:164056
    [221]	validation_0-rmse:163910
    [222]	validation_0-rmse:163769
    [223]	validation_0-rmse:163623
    [224]	validation_0-rmse:163482
    [225]	validation_0-rmse:163337
    [226]	validation_0-rmse:163192
    [227]	validation_0-rmse:163045
    [228]	validation_0-rmse:162900
    [229]	validation_0-rmse:162755
    [230]	validation_0-rmse:162611
    [231]	validation_0-rmse:162471
    [232]	validation_0-rmse:162328
    [233]	validation_0-rmse:162182
    [234]	validation_0-rmse:162039
    [235]	validation_0-rmse:161897
    [236]	validation_0-rmse:161751
    [237]	validation_0-rmse:161609
    [238]	validation_0-rmse:161464
    [239]	validation_0-rmse:161321
    [240]	validation_0-rmse:161180
    [241]	validation_0-rmse:161035
    [242]	validation_0-rmse:160892
    [243]	validation_0-rmse:160749
    [244]	validation_0-rmse:160603
    [245]	validation_0-rmse:160459
    [246]	validation_0-rmse:160316
    [247]	validation_0-rmse:160173
    [248]	validation_0-rmse:160031
    [249]	validation_0-rmse:159889
    [250]	validation_0-rmse:159745
    [251]	validation_0-rmse:159603
    [252]	validation_0-rmse:159461
    [253]	validation_0-rmse:159318
    [254]	validation_0-rmse:159176
    [255]	validation_0-rmse:159034
    [256]	validation_0-rmse:158894
    [257]	validation_0-rmse:158749
    [258]	validation_0-rmse:158608
    [259]	validation_0-rmse:158466
    [260]	validation_0-rmse:158326
    [261]	validation_0-rmse:158186
    [262]	validation_0-rmse:158044
    [263]	validation_0-rmse:157904
    [264]	validation_0-rmse:157765
    [265]	validation_0-rmse:157625
    [266]	validation_0-rmse:157484
    [267]	validation_0-rmse:157341
    [268]	validation_0-rmse:157198
    [269]	validation_0-rmse:157057
    [270]	validation_0-rmse:156921
    [271]	validation_0-rmse:156781
    [272]	validation_0-rmse:156645
    [273]	validation_0-rmse:156507
    [274]	validation_0-rmse:156369
    [275]	validation_0-rmse:156234
    [276]	validation_0-rmse:156095
    [277]	validation_0-rmse:155953
    [278]	validation_0-rmse:155816
    [279]	validation_0-rmse:155678
    [280]	validation_0-rmse:155540
    [281]	validation_0-rmse:155404
    [282]	validation_0-rmse:155267
    [283]	validation_0-rmse:155130
    [284]	validation_0-rmse:154993
    [285]	validation_0-rmse:154855
    [286]	validation_0-rmse:154717
    [287]	validation_0-rmse:154579
    [288]	validation_0-rmse:154442
    [289]	validation_0-rmse:154307
    [290]	validation_0-rmse:154171
    [291]	validation_0-rmse:154032
    [292]	validation_0-rmse:153895
    [293]	validation_0-rmse:153760
    [294]	validation_0-rmse:153623
    [295]	validation_0-rmse:153488
    [296]	validation_0-rmse:153353
    [297]	validation_0-rmse:153217
    [298]	validation_0-rmse:153081
    [299]	validation_0-rmse:152942
    [300]	validation_0-rmse:152808
    [301]	validation_0-rmse:152674
    [302]	validation_0-rmse:152543
    [303]	validation_0-rmse:152408
    [304]	validation_0-rmse:152275
    [305]	validation_0-rmse:152140
    [306]	validation_0-rmse:152007
    [307]	validation_0-rmse:151875
    [308]	validation_0-rmse:151742
    [309]	validation_0-rmse:151609
    [310]	validation_0-rmse:151474
    [311]	validation_0-rmse:151339
    [312]	validation_0-rmse:151207
    [313]	validation_0-rmse:151074
    [314]	validation_0-rmse:150941
    [315]	validation_0-rmse:150810
    [316]	validation_0-rmse:150679
    [317]	validation_0-rmse:150547
    [318]	validation_0-rmse:150417
    [319]	validation_0-rmse:150285
    [320]	validation_0-rmse:150154
    [321]	validation_0-rmse:150022
    [322]	validation_0-rmse:149894
    [323]	validation_0-rmse:149764
    [324]	validation_0-rmse:149629
    [325]	validation_0-rmse:149497
    [326]	validation_0-rmse:149367
    [327]	validation_0-rmse:149235
    [328]	validation_0-rmse:149103
    [329]	validation_0-rmse:148974
    [330]	validation_0-rmse:148843
    [331]	validation_0-rmse:148715
    [332]	validation_0-rmse:148588
    [333]	validation_0-rmse:148458
    [334]	validation_0-rmse:148329
    [335]	validation_0-rmse:148201
    [336]	validation_0-rmse:148071
    [337]	validation_0-rmse:147943
    [338]	validation_0-rmse:147814
    [339]	validation_0-rmse:147686
    [340]	validation_0-rmse:147557
    [341]	validation_0-rmse:147428
    [342]	validation_0-rmse:147299
    [343]	validation_0-rmse:147169
    [344]	validation_0-rmse:147041
    [345]	validation_0-rmse:146911
    [346]	validation_0-rmse:146783
    [347]	validation_0-rmse:146653
    [348]	validation_0-rmse:146524
    [349]	validation_0-rmse:146397
    [350]	validation_0-rmse:146271
    [351]	validation_0-rmse:146139
    [352]	validation_0-rmse:146012
    [353]	validation_0-rmse:145884
    [354]	validation_0-rmse:145758
    [355]	validation_0-rmse:145633
    [356]	validation_0-rmse:145505
    [357]	validation_0-rmse:145378
    [358]	validation_0-rmse:145246
    [359]	validation_0-rmse:145120
    [360]	validation_0-rmse:144992
    [361]	validation_0-rmse:144868
    [362]	validation_0-rmse:144743
    [363]	validation_0-rmse:144617
    [364]	validation_0-rmse:144490
    [365]	validation_0-rmse:144366
    [366]	validation_0-rmse:144240
    [367]	validation_0-rmse:144113
    [368]	validation_0-rmse:143989
    [369]	validation_0-rmse:143863
    [370]	validation_0-rmse:143738
    [371]	validation_0-rmse:143614
    [372]	validation_0-rmse:143489
    [373]	validation_0-rmse:143365
    [374]	validation_0-rmse:143240
    [375]	validation_0-rmse:143115
    [376]	validation_0-rmse:142989
    [377]	validation_0-rmse:142866
    [378]	validation_0-rmse:142741
    [379]	validation_0-rmse:142618
    [380]	validation_0-rmse:142495
    [381]	validation_0-rmse:142375
    [382]	validation_0-rmse:142251
    [383]	validation_0-rmse:142129
    [384]	validation_0-rmse:142007
    [385]	validation_0-rmse:141882
    [386]	validation_0-rmse:141758
    [387]	validation_0-rmse:141635
    [388]	validation_0-rmse:141512
    [389]	validation_0-rmse:141385
    [390]	validation_0-rmse:141264



```
model.feature_importances_
```

Let's see top 10 most important features.


```
plt.figure(figsize=(10, 10))
xgboost.plot_importance(model, max_num_features=10, height=0.2)
```


```

```
