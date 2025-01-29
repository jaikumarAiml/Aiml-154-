#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print


# In[5]:


print(type(data))
print(data.shape)
print(data.size)


# In[6]:


data1=data.drop(['Unnamed: 0',"Temp C"],axis=1)
data1


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep=False)]


# In[9]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[10]:


data1.rename({'Solar.R':'Solar','Day':'Days'},axis=1,inplace=True)
data1


# #### Impute the missing values

# In[12]:


data1.info()


# In[13]:


data1.isnull().sum()


# In[14]:


cols=data1.columns
colors = ['black','Red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[15]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone:",median_ozone)
print("Mean of Ozone:",mean_ozone)


# In[16]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


median_Solar = data1["Solar"].median()
mean_Solar = data1["Solar"].mean()
print("Median of Solar:",median_Solar)
print("Mean of Solar:",mean_Solar)


# In[18]:


data1['Solar']=data1['Solar'].fillna(median_Solar)
data1.isnull().sum()


# In[19]:


# find the mode values of categorical column(Weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


#Impute missing values (Replace NaN with mode etc.)of "Weather using fillna()
data1["Weather"]=data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[21]:


print(data1["Month"].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)
data1["Month"]=data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[22]:


print(data1["Days"].value_counts())
mode_Days = data1["Days"].mode()[0]
print(mode_Days)
data1["Days"]=data1["Days"].fillna(mode_Days)
data1.isnull().sum()


# In[23]:


fig, axes = plt.subplots(2,1, figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Ozone"],ax=axes[0],color='white',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"],kde=True,ax=axes[1],color='brown',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout() 
plt.show()                


#  Observations
# -The ozone column has extreme values beyond 81 as seen from box plot.
# -The same is confirmed from the below right-skewe histogram

# In[25]:


fig, axes = plt.subplots(2,1, figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Solar"],ax=axes[0],color='skyblue',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='green',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout() 
plt.show()  


# Observations
# >There is no outlier in the Solar column.
# >Slightly-left histogram
# >outlier are extreme values in the given data.we find the outlier by the boxplot and histogram

# In[27]:


# Create a figure for violin plot
sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[28]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"],vert = False)


# In[29]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# #### Method 2
# using mu +/-3* sigma limits(standard deviation method)

# In[31]:


data1["Ozone"].describe()


# In[32]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x<(mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# #### Observation
# - it is observed that only two outliers are indenfied using std method
# - in box plot method more no of outliers are identified
# - this is because the assumption of normality is not satified in this column. 

# #### Quantile-Quantile plot for detection of outliers

# In[35]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist="norm",plot=plt)
plt.title("Q-Q Plot for Outlier Detection",fontsize=14)
plt.xlabel("Theoretical Quantiles",fontsize=12)


# #### observation from Q-Q plot
# - The data does not follow normal distribution as the data points are deviating significantly away from the red line
# - The data shows a right-skewed distributation and possible outliers

# In[37]:


#Q-Q plot for Solar
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"],dist="norm",plot=plt)
plt.title("Q-Q Plot for Outlier Detection",fontsize=14)
plt.xlabel("Theoretical Quantiles",fontsize=12)


# In[79]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"],color='pink')
plt.title("violin plot")
plt.show()


# In[83]:


sns.swarmplot(data=data1,x="Weather",y="Ozone",color="orange",palette="Set2",size=6)


# In[89]:


sns.swarmplot(data=data1,x="Ozone",y="Weather",color="green",palette="Set2",size=6)


# In[99]:


sns.stripplot(data=data1,x="Weather",y="Ozone",color="orange",palette="Set1",size=6,jitter=True)


# In[111]:


#kde-kernel density estimate plot
sns.kdeplot(data=data1["Ozone"],fill=True,color="violet")
sns.rugplot(data=data1["Ozone"],color="black")


# In[123]:


sns.boxplot(data=data1,x="Weather",y="Ozone")


# In[135]:


sns.scatterplot(data=data1,x="Weather",y="Ozone")


# #### Correlation coefficient and pair plots

# In[131]:


plt.scatter(data1["Wind"],data1["Temp"])


# In[140]:


#compute pearson correlation coefficient
data1["Wind"].corr(data1["Temp"])


# In[144]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




