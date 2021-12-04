#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyreadstat
import numpy as np 
import pandas as pd 
import os


import seaborn as sns
color = sns.color_palette()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score,recall_score,classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection  import train_test_split


# In[2]:


df = pd.read_sas('C:\Files for python projects\KGB.sas7bdat')


# In[3]:


type(df)


# In[4]:


df.head()


# In[5]:


df['CARDS'].value_counts()


# In[6]:


df['NMBLOAN'].value_counts()


# In[8]:


# let us see build simple plots to see different correlations in this dataset


# In[7]:


df['TITLE'].value_counts()


# In[9]:


sns.lmplot(data=df, x="INCOME", y="CHILDREN", hue="NMBLOAN")


# In[10]:


sns.displot(data=df, x="NMBLOAN")


# In[11]:


sns.boxplot(x=df['NMBLOAN'], y=df['INCOME'], showmeans=True);


# In[12]:


sns.boxplot(x=df['FINLOAN'], y=df['INCOME'], showmeans=True);


# In[13]:


sns.lmplot(x='INCOME',y='GB',data=df, hue='NMBLOAN',fit_reg=False)


# In[14]:


sns.distplot(df.INCOME)


# In[15]:


interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df["AGE"] = pd.cut(df.AGE, interval, labels=cats)


# In[16]:


sns.scatterplot(df.AGE, df.INCOME)


# In[17]:


sns.scatterplot(df.AGE, df.CHILDREN)


# In[18]:


sns.scatterplot(df.CARDS, df.INCOME)
sns.set(rc={'figure.figsize':(11,7,)})


# In[19]:


sns.boxplot(x=df['GB'], y=df['INCOME'], showmeans=True)


# In[20]:


sns.catplot(data=df, x="PERS_H", y="INCOME", hue="NMBLOAN")


# In[21]:


sns.catplot(data=df, x="PERS_H", y="INCOME", hue="GB")


# In[22]:


g=sns.relplot(
    data=df,
    x="INCOME", y="NMBLOAN", hue="CARDS"
)
g.fig.set_size_inches(15,8)


# In[403]:


pip install researchpy


# In[24]:


import researchpy as rp


# In[25]:


rp.ttest(group1= df['NMBLOAN'][df['AGE'] == 'Student'], group1_name= "Student", 
         group2= df['NMBLOAN'][df['AGE'] == 'Young'], group2_name= "Young")


# In[26]:


summary, results = rp.ttest(group1= df['NMBLOAN'][df['AGE'] == 'Student'], group1_name= "Student", 
         group2= df['NMBLOAN'][df['AGE'] == 'Young'], group2_name= "Young")
print(summary)


# In[27]:


print(results)


# In[28]:


import scipy.stats as stats
stats.levene(df['NMBLOAN'][df['AGE'] == 'Student'],
             df['NMBLOAN'][df['AGE'] == 'Young'],
             center= 'mean')


# In[29]:


rp.ttest(group1= df['NMBLOAN'][df['AGE'] == 'Student'], group1_name= "Student", 
         group2= df['NMBLOAN'][df['AGE'] == 'Adult'], group2_name= "Adult")


# In[30]:


rp.ttest(group1= df['NMBLOAN'][df['AGE'] == 'Adult'], group1_name= "Adult", 
         group2= df['NMBLOAN'][df['AGE'] == 'Senior'], group2_name= "Senior")


# In[31]:


summary, results = rp.ttest(group1= df['NMBLOAN'][df['AGE'] == 'Adult'], group1_name= "Adult", 
         group2= df['NMBLOAN'][df['AGE'] == 'Senior'], group2_name= "Senior")


# In[32]:


print(summary)


# In[33]:


print(results)


# In[414]:


# There is no correlation between student/young, student/adult in number of loans. 
# However, there is some correlation between adult/senior in number of loans. 


# In[447]:


# we are going to build logistic regression. 


# In[93]:


df = pd.read_sas('C:\Files for python projects\KGB.sas7bdat')


# In[94]:


df.columns


# In[109]:


target = 'GB'
newdata = ['AGE', 'INCOME', 'NMBLOAN', 'FINLOAN', 'TMJOB1',]


# In[110]:


x=df[newdata].values
y=df[target].values


# In[111]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=428)


# In[112]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
PredictorScaler=MinMaxScaler()

PredictorScalerFit=PredictorScaler.fit(x)
X=PredictorScalerFit.transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[113]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[122]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1,penalty='l2', solver='newton-cg')
LOG=clf.fit(x_train,y_train)
prediction=LOG.predict(x_test)
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

from sklearn.model_selection import cross_val_score
Accuracy_Values=cross_val_score(LOG, X , y, cv=15, scoring='f1_weighted')
print('\nAccuracy values for 15-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




