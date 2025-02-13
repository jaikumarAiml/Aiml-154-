#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install mlxtend library
get_ipython().system('pip install mlxtend')


# In[2]:


# Import necessary libraries
import pandas as pd 
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


#print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# #### Observation
# - In the given DataFrame there are 2201 rows and 4 columns.
# - There are no null values.
# - The given dataframe is object data type and categorical in nature.
# - As the column are categorical, we can adpot one-hot encoding.

# In[6]:


titanic.describe()


# In[7]:


#plot a bar chart to visualize the category of people on the ship
counts=titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# #### Observation
# - There are so many  Crew members in the ship.
# - There are more passengers in the 3rd class.

# In[16]:


counts=titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# #### Observation
# - There are so many male passengers then the female passengers in the ship

# In[18]:


counts=titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# #### Observation
# - There are more Adult passengers in the ship

# In[22]:


counts=titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# #### Observation
# - There are Less passengers are Survived in the ship.

# In[31]:


# Perform onehot encoding on categorical columns
df=pd.get_dummies(titanic,dtype=int)
df.head()


# #### Observation 
# - We change the Categorical data type to int data type.

# In[40]:


df.info()


# #### Apriori Algorithm

# In[43]:


# Apply Apriori algorithm to get itemset combination
frequent_itemsets=apriori(df,min_support=0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[45]:


frequent_itemsets.info()


# In[49]:


# Generate association rules with metrics
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1.0)
rules


# In[52]:


rules.sort_values(by='lift',ascending=True)


# In[56]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




