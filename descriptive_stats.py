#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


#mean value of sat score
np.mean(df["SAT"])


# In[21]:


#median of the data
np.median(df["SAT"])


# In[13]:


#standard deviation of data
np.std(df["GradRate"])


# In[15]:


#find the variance
np.var(df["SFRatio"])


# In[17]:


df.describe()


# In[ ]:




