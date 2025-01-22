#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np


# In[26]:


df = pd.read_csv("Universities.csv")
df


# In[27]:


#mean value of sat score
np.mean(df["SAT"])


# In[28]:


#median of the data
np.median(df["SAT"])


# In[29]:


#standard deviation of data
np.std(df["GradRate"])


# In[30]:


#find the variance
np.var(df["SFRatio"])


# In[31]:


df.describe()


# #### visualization

# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


plt.hist(df["GradRate"])


# In[47]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




