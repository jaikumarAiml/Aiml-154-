#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


#mean value of sat score
np.mean(df["SAT"])


# In[4]:


#median of the data
np.median(df["SAT"])


# In[5]:


#standard deviation of data
np.std(df["GradRate"])


# In[6]:


#find the variance
np.var(df["SFRatio"])


# In[7]:


df.describe()


# #### visualization

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


plt.hist(df["GradRate"])


# In[11]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# #### visualization using boxplot

# In[24]:


s = [20,15,10,25,30,35,28,150,200]
scores = pd.Series(s)
scores


# In[26]:


plt.boxplot(scores, vert=False)


# In[33]:


s = [20,15,10,25,30,35,28,150,200,600,450]
scores = pd.Series(s)
scores


# In[35]:


plt.boxplot(scores, vert=False)


# #### visualization using university data set

# In[39]:


df = pd.read_csv("universities.csv")
df


# In[46]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"], vert=False)


# In[48]:


plt.figure(figsize=(6,2))
plt.title("Box plot for GradRate Score")
plt.boxplot(df["GradRate"], vert=False)


# In[ ]:




