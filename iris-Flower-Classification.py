#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[5]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[7]:


iris.info()


# In[9]:


iris.describe()


# In[11]:


iris.isna().sum()


# In[15]:


iris.duplicated()


# In[21]:


iris[iris.duplicated(keep=False)]


# #### Observation
# - There are no missing values
# - datatypes are float and object
# - There  are no null values
# - There is one duplicated row(101 and 142 are duplicated)
# - all the x-columns are continuous
# - There are three flower cstegories(classes)

# #### Tranform the y-column to categorical using LabelEncoder()

# In[27]:


labelencoder=LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris


# In[ ]:




