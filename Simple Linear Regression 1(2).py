#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1=pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box Plot for Daily Sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[7]:


plt.figure(figsize=(6,3))
plt.title("Box Plot for sunday Sales")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[8]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[9]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# #### Observation
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily columns and also in sunday column as observed from the positive
# 

# #### Scatter plot and Correlation Strength

# In[12]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0,max(x)+100)
plt.ylim(0,max(y)+100)
plt.show()         


# In[13]:


data1["daily"].corr(data1["sunday"])


# In[14]:


data1[["daily","sunday"]].corr()


# In[15]:


data1.corr(numeric_only=True)


# #### Observation on correlation strength
# - The relationship betwwen x(daily) and y(sunday) is seen to ge linear as seen from scatter plot
# - The correlation is strong and positive with Person`s correlation coefficient of 0.958154

# #### Fit a Linear Regression Model

# In[18]:


import statsmodels.formula.api as smf
model1=smf.ols("sunday~daily",data=data1).fit()
model1.summary()


# #### Observation Model Summary
# - The Probability(p-value) for intercept(beta_0)is 0.707 > 0.05.
# - Therefore the intercept coefficient may not that much significant in prediction.
# - However the p-value for "daily"(beta_1) is 0.00 < 0.05.
# - Therfore the beta_1 coefficient is highly signficant and is contributint to prediction.

# #### Interpretation:
# - $R^2$ = 1 perfect fit(all variance explained)
# - $R^2$ = 0 Model does not explain any variance
# - $R^2$ close to 1 $\rightarrow$ Good model fit
# - $R^2$ close to 0 $\rightarrow$ poor model fit

# In[21]:


x=data1["daily"].values
y=data1["sunday"].values
plt.scatter(x,y, color = "m", marker = "o",s=30)
b0=13.84
b1=1.33
y_hat = b0 + b1*x
plt.plot(x,y_hat,color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[22]:


#print the fitted  the coefficient (Beta-0 and Beta-1)
model1.params


# In[23]:


#print the model statistics(t and p-values)
print(f'model t-values:\n{model1.tvalues}\n--------------\nmodel p-values: \n{model1.pvalues}')


# In[24]:


#print the Quality of fitted line for square values
(model1.rsquared,model1.rsquared_adj)


# #### Predict for new data point

# In[26]:


#Predict for 200 and 300 daily circulation
newdata=pd.Series([200,300,1500])


# In[27]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[28]:


model1.predict(data_pred)


# In[29]:


#predict on all given training data
pred = model1.predict(data1["daily"])
pred


# In[30]:


#add predicted values as a column  in data1
data1["Y_hat"]=pred
data1


# In[31]:


#compute the error values (residuals) and add as another columns
data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[32]:


#Compute Mean Squared Error for the the model
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[64]:


mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# #### Assumption in Linear simple linear regression
# 1. **Linearity:** The relationship between the predictors (x) and the response variable (y) is linear.
# 2. **Independence:** Observation are independent of each other.
# 3. **Homoscedasticity:** The residuals (Y-Y_hat) exhibit constant variance at all level of the predictors.
# 4. **Normal Distribution of Error:** The residuals (error) of the model are

# #### Checking the model rediuals scatter plot (for homoscedasticity)

# In[70]:


#plot the residuals versus y_hat
plt.scatter(data1["Y_hat"],data1["residuals"])


# #### Observation
# - There appears to be trend and the residuals are randomly placed around the zero error line
# - Hence the assupmtion of homoscedasticty(constant variance in residuals)is satisfied

# In[80]:


#plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm
sm.qqplot(data1["residuals"],line='45',fit=True)
plt.show()


# In[82]:


#plot the kde distibution foe residuals
sns.histplot(data1["residuals"],kde=True)


# #### Observations:
# - The data points are seen to closely follow the reference line of normality
# - Hence the residuals are approximately normally distributed as also can be seen from the kde distribution
# - All the assumptions of the model are satisfactory and the model is accurated

# In[ ]:




