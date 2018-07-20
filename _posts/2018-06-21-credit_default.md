---
layout: post
title: "Taiwanese Credit Analysis with Tensorflow"
permalink: /taiwanese-credit/
---

# Taiwanese Credit Default Analysis


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import tflearn
import tensorflow as tf
```

## Data Import and Cleaning

This notebook uses a Tiawanese credit card dataset provided by the University of California Irvine. Let's begin by importing our data downloaded at `https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset#UCI_Credit_Card.csv`.


```python
df_credit = pd.read_csv('/Users/tnightengale/Desktop/Kaggle/Credit_Fraud/UCI_Credit_Card.csv')
```

Let's use pandas to checkout the data and beging the process of cleaning it. We'll start by examining the top entries of the dataset, as well as the documentation for the included features.


```python
df_credit.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



The documentation for the dataset is presented below:

- ID: ID of each client
- LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
- SEX: Gender (1=male, 2=female)
- EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- MARRIAGE: Marital status (1=married, 2=single, 3=others)
- AGE: Age in years
- PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two -months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
- PAY_2: Repayment status in August, 2005 (scale same as above)
- PAY_3: Repayment status in July, 2005 (scale same as above)
- PAY_4: Repayment status in June, 2005 (scale same as above)
- PAY_5: Repayment status in May, 2005 (scale same as above)
- PAY_6: Repayment status in April, 2005 (scale same as above)
- BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
- BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
- BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
- BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
- BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
- BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
- PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
- PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
- PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
- PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
- PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
- PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)

The binary variables of `SEX` and `MARRIAGE` make more sense as dummy variables coded as `0` or `1` and renamed `MALE` and `MARRIED`. Let's rename and reformat those variables. The intuition is that having a sparse vector for a column will likely speed up computation time for our neural network.


```python
# rename and format `SEX` column
mask = df_credit.SEX == 2
column_name = 'SEX'
df_credit.loc[mask, column_name] = 0

df_credit.SEX = df_credit.rename(columns = {'SEX':'MALE'}, inplace = True)
```

Let's make sure the labels for `MARRIAGE` make sense. 


```python
df_credit.MARRIAGE.value_counts()
```




    2    15964
    1    13659
    3      323
    0       54
    Name: MARRIAGE, dtype: int64



Clearly there are some missing values here, being coded as `3` and `0`. Let's switch them to NaN and check the result.


```python
df_credit.MARRIAGE = df_credit['MARRIAGE'].replace({0:np.nan, 3:np.nan})
df_credit = df_credit.dropna()
```


```python
df_credit.MARRIAGE.value_counts()
```




    2.0    15964
    1.0    13659
    Name: MARRIAGE, dtype: int64



Let's rename the `MARRIAGE` column.


```python
# rename and format `MARRIAGE` column
mask = df_credit.MARRIAGE == 2
column_name = 'MARRIAGE'
df_credit.loc[mask, column_name] = 0

df_credit.MARRIAGE = df_credit.rename(columns = {'MARRIAGE':'MARRIED'}, inplace = True)
```

Let's check out our reformatted labels and binary variables.


```python
pd.set_option('display.max_columns', 500)
df_credit.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>MALE</th>
      <th>EDUCATION</th>
      <th>MARRIED</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000.0</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913.0</td>
      <td>3102.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682.0</td>
      <td>1725.0</td>
      <td>2682.0</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239.0</td>
      <td>14027.0</td>
      <td>13559.0</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000.0</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990.0</td>
      <td>48233.0</td>
      <td>49291.0</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617.0</td>
      <td>5670.0</td>
      <td>35835.0</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>50000.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>64400.0</td>
      <td>57069.0</td>
      <td>57608.0</td>
      <td>19394.0</td>
      <td>19619.0</td>
      <td>20024.0</td>
      <td>2500.0</td>
      <td>1815.0</td>
      <td>657.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>800.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>500000.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>367965.0</td>
      <td>412023.0</td>
      <td>445007.0</td>
      <td>542653.0</td>
      <td>483003.0</td>
      <td>473944.0</td>
      <td>55000.0</td>
      <td>40000.0</td>
      <td>38000.0</td>
      <td>20239.0</td>
      <td>13750.0</td>
      <td>13770.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>100000.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>23</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>11876.0</td>
      <td>380.0</td>
      <td>601.0</td>
      <td>221.0</td>
      <td>-159.0</td>
      <td>567.0</td>
      <td>380.0</td>
      <td>601.0</td>
      <td>0.0</td>
      <td>581.0</td>
      <td>1687.0</td>
      <td>1542.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>140000.0</td>
      <td>0</td>
      <td>3</td>
      <td>1.0</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11285.0</td>
      <td>14096.0</td>
      <td>12108.0</td>
      <td>12211.0</td>
      <td>11793.0</td>
      <td>3719.0</td>
      <td>3329.0</td>
      <td>0.0</td>
      <td>432.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>20000.0</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>35</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13007.0</td>
      <td>13912.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13007.0</td>
      <td>1122.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Great. Now let's check out the values for the `PAY_` features.


```python
df_credit['PAY_0'].value_counts()
```




     0    14532
    -1     5629
     1     3635
    -2     2738
     2     2634
     3      315
     4       76
     5       25
     8       19
     6       11
     7        9
    Name: PAY_0, dtype: int64



These features seem somewhat ambiguous. It's unclear how the categorical designations for the various payment columns work. The quality of this dataset is uncertain: it's publically available with little documentation. Credit data by nature is sensitive. Therefore quality credit data is usually confidentially held by firms.

Unfortunately this is the data we have to work with. Let's see what type of predictive accuracy we are able to coax out of it: we're ready to start predicting credit default using these features.

## Building the Model

Attempt to build a simple 3 layer classification fully-connected model.

Let's first save our edited dataframe as a csv. Then reload it using `tflearn`.


```python
tf.reset_default_graph()
```


```python
df_credit.to_csv('/Users/tnightengale/Desktop/Kaggle/Credit_Fraud/cleaned_credit_data.csv')
```


```python
# check that csv worked as intended
df_check = pd.read_csv('/Users/tnightengale/Desktop/Kaggle/Credit_Fraud/cleaned_credit_data.csv')
```


```python
df_check.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>MALE</th>
      <th>EDUCATION</th>
      <th>MARRIED</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>3913.0</td>
      <td>3102.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>120000.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2682.0</td>
      <td>1725.0</td>
      <td>2682.0</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>90000.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29239.0</td>
      <td>14027.0</td>
      <td>13559.0</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>50000.0</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46990.0</td>
      <td>48233.0</td>
      <td>49291.0</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8617.0</td>
      <td>5670.0</td>
      <td>35835.0</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We're going to use the `tflearn` library to quickly experiment with a couple of fully-connected network configurations. `tflearn` is a high-level API for tensorflow, that allows us to quickly construct layers, choose a cost function, and interject various normalization transformations into our network.


```python
data, target = tflearn.data_utils.load_csv('/Users/tnightengale/Desktop/Kaggle/Credit_Fraud/cleaned_credit_data.csv',
                                          categorical_labels = True, n_classes=2)
```


```python
# preprocess the data 
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    return np.array(data, dtype=np.float32)

to_ignore = [0,1]

data = preprocess(data, to_ignore)
```

Below the layers of the network are built recursively, taking in the pervious layer as input at each new layer.


```python
n_features = len(df_credit.columns[1:len(df_credit.columns)-1])
net = tflearn.input_data(shape=[None, n_features])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
```


```python
# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, target, n_epoch=10, batch_size=16, show_metric=True)
```

    Training Step: 18519  | total loss: [1m[32m0.63909[0m[0m | time: 5.017s
    | Adam | epoch: 010 | loss: 0.63909 - acc: 0.7168 -- iter: 29616/29623
    Training Step: 18520  | total loss: [1m[32m0.61910[0m[0m | time: 5.019s
    | Adam | epoch: 010 | loss: 0.61910 - acc: 0.7264 -- iter: 29623/29623
    --


Hmmm our accuracy of 0.72 is pretty weak. Let's implement a relu activation in our first two fully connected layers, as well as a dropout mechanism to prevent overfitting and compare our accuracy:


```python
tf.reset_default_graph()
```


```python
net_1 = net = tflearn.input_data(shape=[None, n_features])
net_1 = tflearn.fully_connected(net_1, 32, activation='relu')
net_1 = tflearn.dropout(net_1, 0.8)
net_1 = tflearn.fully_connected(net, 32, activation='relu')
net_1 = tflearn.dropout(net_1, 0.8)
net_1 = tflearn.fully_connected(net_1, 2, activation='softmax')
net_1 = tflearn.regression(net_1)
```


```python
# new model with relu and dropout model
model = tflearn.DNN(net_1)
# Start training (apply gradient descent algorithm)
model.fit(data, target, n_epoch=10, batch_size=16, show_metric=True)
```

    Training Step: 18519  | total loss: [1m[32m5.28767[0m[0m | time: 4.598s
    | Adam | epoch: 010 | loss: 5.28767 - acc: 0.7704 -- iter: 29616/29623
    Training Step: 18520  | total loss: [1m[32m4.90281[0m[0m | time: 4.601s
    | Adam | epoch: 010 | loss: 4.90281 - acc: 0.7871 -- iter: 29623/29623
    --


Looks like the relu activations impart a higher training accuracy. Let's build out a deeper network, whilst still using our dropout mechanism to reduce overfitting. We will also adjust our cost function to better suit the nature of the credit default problem.

What would be more costly to a credit firm: classifying a paying customer as defaulting (false postive) or classifying a defaulting customer as paying (false negative)? In the case of a false positive, the customers who will pay may receive a signal, a notice for example, indicating that they may be at risk for higher interest rates if they do not comply and pay their balance. This may annoy them slightly, but if they intend to pay their balance anyway, it will likely not be a significant issue. However, in the case of a false negative, the customers who will default will not receive any signal to alter their behaviour and their default will constrain the firm's cashflows. Therefore, the firm will likely inccur more costs by classifying a defaulting customer as paying.

To account for this, let's implement a weighted loss function, that disporportionately penalizes false negative results.

defaulting is cancer. test says no cancer. patient has cancer. => test says no default. person defaults. => false negative

Our weighted crossentropy function is as follows:

$$
\mathcal{Loss} = -\frac{1}{m} \sum\limits_{i = 1}^{m} \bigg(\mathcal{W}\cdot(y^{(i)}\log\left(\sigma(\hat{y}^{i})\right) + (1-y^{(i)})\log\left(1- \sigma(\hat{y}^{i}\right)\bigg)
$$

where $\sigma(\hat{y})$ is the sigmoid/logistic function and $\mathcal{W}$ is the relative weighting between positive and negative errors.

Recall that we are trying to predict which customers will default. Default is indicated in our dataset using `1` whereas customers in good financial standing are labelled `0`. Therefore we want $\mathcal{W} > 1$ porportional to the relatively higher cost of not predicting a defaulting customer.


```python
tf.reset_default_graph()
```


```python
net_2 = tflearn.input_data(shape=[None, n_features])
net_2 = tflearn.fully_connected(net_2, 16, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 32, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 64, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 256, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 128, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 128, activation='relu')
net_2 = tflearn.dropout(net_2, 0.8)
net_2 = tflearn.fully_connected(net_2, 2, activation='softmax')
net_2 = tflearn.regression(net_2,loss=lambda data, target: weighted_crossentropy(data,target,weight=3))
# model 2 with relu and dropout model
model = tflearn.DNN(net_2, best_val_accuracy=.8, best_checkpoint_path='/Users/tnightengale/Desktop/Kaggle/Credit_Fraud')
# Start training (apply gradient descent algorithm)
model.fit(data, target, n_epoch=10, batch_size=16, show_metric=True, validation_set=0.1, snapshot_epoch=True)
```

    Training Step: 16669  | total loss: [1m[32m0.97229[0m[0m | time: 7.524s
    | Adam | epoch: 010 | loss: 0.97229 - acc: 0.8229 -- iter: 26656/26660
    Training Step: 16670  | total loss: [1m[32m0.96221[0m[0m | time: 8.694s
    | Adam | epoch: 010 | loss: 0.96221 - acc: 0.8344 | val_loss: 1.01543 - val_acc: 0.7739 -- iter: 26660/26660
    --


## Conclusion

Our model acheives a training accuracy of 0.83 and a test accuracy of approximately 0.77. While this is obviously better than simply guessing defaulting customers, this is insufficient to the current needs of modern firms concerned with credit default.

The take-aways of this notebook are:
- pandas can be used as a preliminary cleaning tool
- tflearn is a great library for trying out different neural network configurations quickly
- approaching problems critically is essential to obtain reasonable solutions

In our case the cost of failing to detect a defauting customer is likely higher than incorrectly labelling a paying customer. We applied a weighted loss function to account for this imbalance in the types of errors made by the model.
