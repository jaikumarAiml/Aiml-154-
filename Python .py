#!/usr/bin/env python
# coding: utf-8

# In[27]:


L=[100,50,193,443,256,53,1000]
L1=[]
L2=[]
for x in L:
    if x%2 == 0:
        L1.append(x)
    else:
        L2.append(x)
print("The given elements are",L)
print("Even numbers",L1)
print("Odd numbers",L2)


# In[38]:


L = ["A", 100, "B", 20.5, True, 6+8j]
L1 = []
L2 = []
L3 = []
L4 = []
L5 = []
for item in L:
    if type(item) == str:
        L1.append(item)
    elif type(item) == int:
        L2.append(item)
    elif type(item) == float:
        L3.append(item)
    elif type(item) == complex:
        L4.append(item)
    elif type(item) == bool:
        L5.append(item)
print("Strings:", L1)
print("Integers:", L2)
print("Floats:", L3)
print("Complex Numbers:", L4)
print("Booleans:", L5)


# In[1]:


#heterogeneous data types
my_list = [1,"hello",3.14,[4,5]]
print(my_list)


# In[3]:


#indexing and slicing
my_list = [10,20,30,40,50]
print(my_list[1])
print(my_list[1:3])
print(my_list[1:])
print(my_list[:3])
print(my_list[:])


# In[4]:


#iterable
my_list=[10,20,30,40]
for item in my_list:
    print(3*item)


# In[10]:


my_list = [10,20,30,40]
for x in range(len(my_list)):
    print(3*my_list[x])


# In[11]:


#duplicate value
duplicates_list=[1,2,3,4,4,3]
duplicates_list


# In[12]:


#Support Nesting
nested_list = [[1,2],[3,4],[5,6]]
nested_list


# In[13]:


dir(list)


# In[14]:


#Initial list of orders:[item_name,quantity]
orders = [["apple",10],["banana",5],["cherry",7],["banana",5]]
orders


# In[17]:


#add a new order with append()
new_orders1 = ["data",12]
orders.append(new_orders1)
orders


# In[21]:


#add multiple order at a time using extend
new_orders = [["grape",9],["apple",20]]
orders.extend(new_orders)
orders


# In[22]:


#add (insert) an order at an index position 3
new_order2 = ["dragon",10]
orders.insert(3,new_order2)
orders


# In[23]:


#remove a duplicate order with using pop
duplicate_order=["banana",5]
orders.pop(4)
orders


# In[ ]:


#order list
[["apple",10],["banana",5],["cherry",7],["dragon",10],["grape",9],["apple",20]]


# In[1]:


#update the quantity of an item("cherry",10)
orders =[["apple",10],["banana",5],["cherry",7],["dragon",10],["grape",9],["apple",20]]
for each in orders:
    if each[0] == "cherry":
        each[1] = 10
        print(orders)


# In[29]:


orders=[["apple",10],["banana",5],["cherry",7],["dragon",10],["grape",9],["apple",20]]
apple_orders = []
for each in orders:
    if each[0] == "apple":
        apple_orders.append(each[1])
print(apple_orders)
print(sum(apple_orders))


# In[33]:


Tup1=(3,5,9,10)
Tup2=10,20,30,"hi"
Tup3=tuple([1,2,5,7])
print(Tup1)
print(Tup2)
print(Tup3)


# In[31]:


tup1 = (4,10,9,"A",9.81,False,9-6j)
print(tup1)
print(type(tup1))


# In[36]:


tup4=([3,4,5],(1,3,6),"Hello",10.5)
print(tup4)


# In[39]:


tup4=([1,2,3],(4,8,9),"Hello",9.8)
tup4[1][1]=100
print(tup4)


# In[40]:


tup4=([1,2,3],(4,8,9),"Hello",9.8)
tup4[0][2]=100
print(tup4)


# In[48]:


#use index() method to find the index value
tup4=([1,2,3],(4,8,9),"Hello",9.8)
tup4.index("Hello")


# In[45]:


tup5=(10,20,1,10,30,10)
tup5.count(10)


# In[ ]:




