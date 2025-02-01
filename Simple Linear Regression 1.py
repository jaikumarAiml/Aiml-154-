#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1=pd.read_csv("NewspaperData.csv")
data1


# In[6]:


data1.info()


# In[8]:


data1.isnull().sum()


# In[10]:


data1.describe()


# In[12]:


plt.figure(figsize=(6,3))
plt.title("Box Plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[14]:


plt.figure(figsize=(6,3))
plt.title("Box Plot for sunday Sales")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[16]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[18]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# #### Observation
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily columns and also in sunday column as observed from the positive
# 

# #### Scatter plot and Correlation Strength

# In[24]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0,max(x)+100)
plt.ylim(0,max(y)+100)
plt.show()         


# In[27]:


data1["daily"].corr(data1["sunday"])


# In[29]:


data1[["daily","sunday"]].corr()


# In[31]:


data1.corr(numeric_only=True)


# #### Observation on correlation strength
# - The relationship betwwen x(daily) and y(sunday) is seen to ge linear as seen from scatter plot
# - The correlation is strong and positive with Person`s correlation coefficient of 0.958154

# #### Fit a Linear Regression Model

# In[45]:


import statsmodels.formula.api as smf
model1=smf.ols("sunday~daily",data=data1).fit()
model1.summary()


# In[ ]:




