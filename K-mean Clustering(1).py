#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# #### Clustering- Divide the universities in to group(Cluster)

# In[3]:


univ=pd.read_csv("Universities.csv")
univ


# In[4]:


univ.info()


# In[5]:


univ.isnull().sum()


# In[6]:


univ.describe()


# In[7]:


univ.boxplot()


# In[8]:


import seaborn as sns
sns.kdeplot(univ=["SAT"], fill=True, color="red")
plt.show()


# #### Standardization of the data

# In[10]:


# Read all numeric columns in to univ1
Univ1=univ.iloc[:,1:]
Univ1


# In[11]:


cols=Univ1.columns
cols


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[13]:


# Build 3 clusters using KMeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[14]:


#print the cluster labels
clusters_new.labels_


# In[15]:


set(clusters_new.labels_)


# In[25]:


#Assign clusters to the Univ data set
univ['clusterid_new']=clusters_new.labels_
univ


# In[27]:


univ.sort_values(by = "clusterid_new")


# In[29]:


univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### Observation
# - Custer 2 appears to be the top rated universities cluster as the cut off score.Top10,SFRatio parameter mean values are highest.
# - Cluster 1 appears to occupy the midle level rated universities.
# - Cluster 0 comes as the lower level rated universities.
# - The Top Rated cluster is cluserid_new(2).
# - The Second top rated cluster is 1.

# In[33]:


univ[univ['clusterid_new']==0]


# #### Finding optimal k value using elbow plot

# In[56]:


wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show ()


# #### Observation
# - From above graph we can choose  k= 3 0r 4 which indicate elbow joins,i.e the rate of change of flow decrease 

# #### Clustering Methods:
#  1. Hierarchical clustering
#  2. K-means clustering
#  3. K-medoids clustering
#  4. K-prototype clustering
#  5. DBSCAN clustering

# In[ ]:




