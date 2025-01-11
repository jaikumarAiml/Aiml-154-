#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create pandas series
import pandas  as pd
data =  [10,20,30,40]
series=pd.Series(data)
print(series)


# In[2]:


#create series using a custom index
data = [1,2,3,4]
i=['A','B','C','D']
series = pd.Series(data,index=i)
print(series)


# In[3]:


#create pandas series using dictionary
data = {'a' :10,'b': 20,'c' : 30}
series = pd.Series(data)
print(series)


# pandas dataframe

# In[5]:


series.replace(20,100)


# In[6]:


#create series using numpy array
import numpy as np
data = np.array([100,200,300])
series = pd.Series(data,index=['a','b','c'])
print(series)


# In[7]:


data = [[1,'Alice',25],[2,'Bob',30],[3,'Mary',34]]
print(data)
df=pd.DataFrame(data,columns=['ID','Name','Age'])
print(df)


# In[8]:


iris_df = pd.read_csv("iris.csv")
print(iris_df)


# In[9]:


#create pandas dataframe from dictionary of lists
import pandas as pd
data = {'Name':['Alice','Bob','Mary'],'Age':[25,30,34],'Country':["USA","UK","AUS"]}
df = pd.DataFrame(data)
print(df)
    


# In[32]:


iris_df = pd.read_excel("iris.xlsx")
print(iris_df)


# In[ ]:





# In[ ]:





# In[10]:


#create pandas dataframe from numpy array
import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df = pd.DataFrame(array, columns=['A','B','C'])
print(df)


# In[ ]:




