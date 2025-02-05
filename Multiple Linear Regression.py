#!/usr/bin/env python
# coding: utf-8

# #### Assumption in Multi linear Regression
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear
# 2. Independence: Observation are independent of each other
# 3. Homoscedasticity: The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Erroe: The residuals of the model are normally distributed
# 5. No multicollinearity: The independent variable should not be too highly correlated with each other.
# - Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns
# - MPG: Milege of the car(Mile per Gallon)(This is T-column to be predicted)
# - HP:Horse Power of the car(X1 column)
# - VOL:Volume of the car(size)(X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 column)
# - WT:Weight of the car(pounds)(X4 column)

# #### EDA

# In[12]:


cars.info()


# In[17]:


#check for missing values.
cars.isna().sum()


# #### Observations about info(),missing values
# - There are no missing values
# - There are 81 observation (81 diffrent cars data)
# - The data type of the columns are also relevant and valid

# In[23]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[25]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[27]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[29]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='MPG',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='MPG',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[31]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# #### Observation from boxplot and histogram
# - These are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# - In VOL and WT columns,a few outliers are observed in both tails of their distribution
# - The extreme values of cars data may hava come from the specially designed nature of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may to be considered while building the regression model

# #### Checking for duplicated rows

# In[35]:


cars[cars.duplicated()]


# #### Pairs plots and Correlation Coefficients

# In[38]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[42]:


cars.corr()


# #### Observations
# - The highest correlation coefficient is  between HP and Sp is(0.973848)
# - The next Highest correlation coefficient is between VOL and VOL(1.000000)

# In[ ]:




