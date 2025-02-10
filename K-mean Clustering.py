#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# #### Clustering- Divide the universities in to group(Cluster)

# In[6]:


univ=pd.read_csv("Universities.csv")
univ


# In[10]:


univ.info()


# In[14]:


univ.isnull().sum()


# In[16]:


univ.describe()


# In[18]:


univ.boxplot()


# In[26]:


import seaborn as sns
sns.kdeplot(univ=["SAT"], fill=True, color="red")
plt.show()


# In[50]:


# Read all numeric columns in to univ1
Univ1=univ.iloc[:,1:]
Univ1


# In[62]:


cols=Univ1.columns
cols


# In[54]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[ ]:




