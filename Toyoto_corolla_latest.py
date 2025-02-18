#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np


# In[4]:


cars = pd.read_csv("Toyoto_corrola.csv")
cars.head()


# In[6]:


cars.info()


# In[8]:


cars.describe()


# In[10]:


#check for missing values.
cars.isna().sum()


# In[12]:


cars[cars.duplicated()]


# #### Observation
# - There are no null values.
# - There is no missing values.
# - In the above dataset [dtypes: int64(9), object(1)]
# - There are 5 continuous columns and 3 categorical columns

# In[25]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='Price',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='Price',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[40]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='Age_08_04',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='Age_08_04',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[42]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='KM',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='KM',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[46]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[52]:


counts = cars["Gears"].value_counts().reset_index()
print(counts)


# In[61]:


counts=cars["Gears"].value_counts()
plt.bar(counts.index,counts.values)


# In[63]:


counts=cars["Cylinders"].value_counts()
plt.bar(counts.index,counts.values)


# In[ ]:




