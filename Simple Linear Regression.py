#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[5]:


data1=pd.read_csv("NewspaperData.csv")
data1


# In[7]:


data1.info()


# In[9]:


data1.describe()


# In[11]:


data1.boxplot()


# In[45]:


sns.kdeplot(data=data1["daily"], fill=True, color="red")
sns.rugplot(data=data1["sunday"], color="black")
plt.show()


# In[49]:


sns.scatterplot(data=data1, x="daily", y="sunday", color="green")
plt.title("Scatter Plot of Daily vs Sunday")
plt.xlabel("Daily")
plt.ylabel("Sunday")
plt.show()


# In[60]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[62]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# #### Correlation

# In[65]:


data1["daily"].corr(data1["sunday"])


# In[67]:


data1[["daily","sunday"]].corr()


# In[69]:


data1.corr(numeric_only=True)


# In[71]:


data1.corr(numeric_only=True)


# #### Fitting a Linear Regression Model

# In[76]:


# Build regression model
import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()
model.summary()


# In[78]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
 
# plotting the regression line
plt.plot(x, y_hat, color = "g")
  
# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




