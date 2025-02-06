#!/usr/bin/env python
# coding: utf-8

# #### Assumption in Multi linear Regression
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear
# 2. Independence: Observation are independent of each other
# 3. Homoscedasticity: The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Erroe: The residuals of the model are normally distributed
# 5. No multicollinearity: The independent variable should not be too highly correlated with each other.
# - Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns
# - MPG: Milege of the car(Mile per Gallon)(This is T-column to be predicted)
# - HP:Horse Power of the car(X1 column)
# - VOL:Volume of the car(size)(X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 column)
# - WT:Weight of the car(pounds)(X4 column)

# #### EDA

# In[7]:


cars.info()


# In[8]:


#check for missing values.
cars.isna().sum()


# #### Observations about info(),missing values
# - There are no missing values
# - There are 81 observation (81 diffrent cars data)
# - The data type of the columns are also relevant and valid

# In[10]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[13]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='MPG',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='MPG',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[14]:


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

# In[17]:


cars[cars.duplicated()]


# #### Pairs plots and Correlation Coefficients

# In[19]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[20]:


cars.corr()


# #### Observations
# - The highest correlation coefficient is  between HP and Sp is(0.973848)
# - The next Highest correlation coefficient is between VOL and VOL(1.000000)
# - Between x and y.all the x variable are showing moderste to high correlation strength, highest being between HP and MPG.
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3 and x4),some very high correlation strength are observed between SP vs HP,VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem 

# #### Preparing a preliminary model considering all X columns

# In[23]:


import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[24]:


model1.summary()


# #### Observation from model summary
# - The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is colse to zero,undicating that all or some of X columns are signficant
# - The p-value for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# #### Performance metric for model1 (By Mean Squared Error)

# In[74]:


#Find the performance metrics
#create a data frame with actual y and predicated y columns
df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[76]:


pred_y1=model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[ ]:




