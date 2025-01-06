#!/usr/bin/env python
# coding: utf-8

# In[1]:


my_list = [10, 20, 30, 40, 50]
print("List - First element:", my_list[0])
print("List - Last element:", my_list[-1])
print("List - First three elements:", my_list[:3])

my_tuple = ('apple', 'banana', 'cherry', 'date', 'elderberry')
print("Tuple - Second element:", my_tuple[1])
print("Tuple - Last element:", my_tuple[-1])
print("Tuple - Middle elements:", my_tuple[1:4])

my_dict = {
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "profession": "Engineer",
    "hobby": "Painting"
}
print("Dictionary - Name:", my_dict["name"])
print("Dictionary - Profession:", my_dict["profession"])
print("Dictionary - Hobby (safe access):", my_dict.get("hobby"))
print("Dictionary - Country (default):", my_dict.get("country", "Not Found"))


# In[1]:


marks1 = float(input("Enter marks for subject 1: "))
marks2 = float(input("Enter marks for subject 2: "))
marks3 = float(input("Enter marks for subject 3: "))

average = (marks1 + marks2 + marks3) / 3

if average >= 90:
    print("Grade: A")
elif 80 <= average < 90:
    print("Grade: B")
elif 70 <= average < 80:
    print("Grade: C")
else:
    print("Grade: Fail")


# In[ ]:




