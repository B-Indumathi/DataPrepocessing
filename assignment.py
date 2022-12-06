#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 


# In[28]:


dataset = pd.read_csv("Flowers.csv")


# In[29]:


dataset


# In[30]:


cols = ['Seller','Name']
dataset = dataset.drop(cols, axis=1)


# In[31]:


dataset.info()


# In[43]:


dataset.isna().sum()


# In[44]:


from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
state_encoded=le.fit_transform(dataset['Family'])
dataset['Family'] = state_encoded
dataset.head()


# In[45]:


X = dataset.iloc[:, :-1].values
X


# In[35]:


Y = dataset.iloc[:, -1].values 
Y


# In[46]:


dataset.describe()


# In[50]:


dataset['Count'] = dataset['Count'].fillna(dataset['Count'].count())


# In[51]:


dataset.isna().sum()


# In[52]:


dataset['Price'] = dataset['Price'].fillna(dataset['Price'].min())


# In[53]:


dataset['LastingDays'] = dataset['LastingDays'].fillna(dataset['LastingDays'].mean())


# In[54]:


dataset.isna().sum()


# In[55]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[56]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_train = sc_X.transform(X_test)
X_train


# In[ ]:




