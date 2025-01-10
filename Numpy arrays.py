#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import numpy library 
import numpy as np


# In[5]:


#create 1D numpy array
x=np.array([45,67,57,60])
print(x)
print(type(x))
print(x.dtype)


# In[7]:


#create 1D numpy array
x=np.array([45,67,57,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[11]:


#create 1D numpy array
x=np.array([45,'A',67,57,60,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[15]:


a2=np.array([[20,40],[30,60]])
print(a2)
print(type(a2))
print(a2.shape)


# In[17]:


#Reshapeing an array using reshape function()
a=np.array([10,20,30,40])
b=a.reshape(2,2)
print(b)
print(b.shape)


# In[19]:


#create an array with arange()
c=np.arange(3,10)
print(c)
type(c)


# In[27]:


#use of around()
d=np.array([1.3467,3.10987,4.91235])
print(d)
np.around(d,2)


# In[31]:


#use of np.sqrt()
d=np.array([1.3467,3.10987,4.91235])
print(d)
print(np.around(np.sqrt(d),2))


# In[43]:


#create a 2d array
a1=np.array([[3,4,5,8],[7,2,8,np.NaN]])#nan=missing data(not a number)
print(a1)
a1.dtype


# In[49]:


#use of astype() 
a1=np.array([[3,4,5,8],[7,2,8,np.NaN]])#nan=missing data(not a number)
a1_copy1=a1.astype(str)
print(a1_copy1)
a1_copy1.dtype


# In[51]:


#Mathematical operations on rows and col
a2=np.array([[3,4,6],[7,9,10],[4,6,12]])
a2


# In[57]:


a2.sum(axis=1)


# In[60]:


a2.sum(axis=0)


# In[62]:


np.sqrt(a2)


# In[64]:


#Matrix operation
a3=np.array([[3,4,5],[7,2,8],[9,1,6]])
a3


# In[74]:


np.fill_diagonal(a3,0) 
print(a3)


# In[84]:


#perform matrix multiplication
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
c=np.matmul(A,B)
c


# In[90]:


#transpose 
print(A.T)
print(B.T)


# In[92]:


#accessing the array element
a4=np.array([[3,4,5],[7,2,8],[9,1,6],[10,9,18]])
a4


# In[98]:


a4[2][2]


# In[100]:


a4[2][0]


# In[102]:


a4[1:3,0:2]


# In[104]:


a4[1:3,0:3]


# In[106]:


a4[0:3,2:3]


# In[118]:


#accessing row max value and its index
a3=np.array([[3,4,5],[7,2,8],[9,1,6]])
print(a3)
print()
print(np.argmax(a3,axis=1))
print(np.argmax(a3,axis=0))    


# In[120]:


#print the max value elements
print(np.amax(a3,axis=1))
print(np.amax(a3,axis=0))


# In[ ]:




