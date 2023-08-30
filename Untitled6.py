#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[10]:


df = pd.read_csv("C:/Users/DCL\Desktop\ML\Data\cars.csv")
df


# In[11]:


df=df.iloc[:,:]
df


# In[16]:


df[['brand','fuel','owner']]


# In[24]:


df['brand'].unique()


# In[20]:


df['owner'].unique()


# In[ ]:





# In[22]:


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault','Mahindra', 'Tata', 'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW','Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo','Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot'],['Diesel', 'Petrol', 'LPG', 'CNG'],['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car']])
df[['brand','fuel','owner']]= oe.fit_transform(df[['brand','fuel','owner']])


# In[23]:


df[['brand','fuel','owner']]


# In[26]:


X= df.drop('selling_price', axis=1)
y= df['selling_price']


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
X_train.shape, X_test.shape


# In[28]:


#x' = (x- mean)/std

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# transform train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[29]:


scaler.mean_


# In[30]:


X_train


# In[31]:


X_train_scaled


# In[32]:


X_train_scaled.dtype


# In[33]:


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[35]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.scatter(X_train['km_driven'], X_train['fuel'])
plt.xlabel('km_driven')
plt.ylabel('fuel')
plt.title('Before Scaling')

plt.subplot(1,2,2)
plt.scatter(X_train_scaled['km_driven'], X_train_scaled['fuel'],color='red')
plt.xlabel('km_driven')
plt.ylabel('fuel')
plt.title('After Scaling')


plt.show()


# In[36]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(X_train['km_driven'])
plt.title('Before Scaling')

plt.subplot(1,2,2)
sns.distplot(X_train_scaled['km_driven'])
plt.title('After Scaling')

plt.show()


# In[38]:


df['km_driven'].skew()


# In[40]:


print("Mean value of km_driven",df['km_driven'].mean())
print("Std value of km_driven",df['km_driven'].std())
print("Min value of km_driven",df['km_driven'].min())
print("Max value of km_driven",df['km_driven'].max())


# In[43]:


# Finding the boundary values
upper_limit= df['km_driven'].mean() + 3*df['km_driven'].std()
lower_limit= df['km_driven'].mean() - 3*df['km_driven'].std()


# In[45]:


# Finding outlier
outlier = df[(df['km_driven'] > upper_limit) | (df['km_driven'] < lower_limit)]
outlier


# In[46]:


# Removing outlier
new_df = df[(df['km_driven'] < upper_limit) & (df['km_driven'] > lower_limit)]
new_df.shape


# In[50]:


# Finding the z score
df['km_driven'] = (df['km_driven'] - df['km_driven'].mean())/df['km_driven'].std()


# In[51]:


# Finding outlier
outlier = df[(df['km_driven'] > 3) | (df['km_driven'] < -3)]
outlier


# In[52]:


new_df1 = df[(df['kmdriven_zscore'] < 3 ) & (df['kmdriven_zscore'] > -3)]
new_df1.shape


# In[53]:


# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['km_driven'])

plt.subplot(2,2,2)
sns.boxplot(df['km_driven'])

plt.subplot(2,2,3)
sns.distplot(new_df1['km_driven'])

plt.subplot(2,2,4)
sns.boxplot(new_df1['km_driven'])


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.preprocessing import PowerTransformer


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[56]:


# Plotting the distplots without any transformation

for col in X_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()


# In[57]:


# Applying Regression without any transformation
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

r2_score(y_test,y_pred)


# In[58]:


import numpy as np
import warnings
np.warnings = warnings


# In[59]:


# Applying Box-Cox Transform

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
X_train_transformed = pt.fit_transform(X_train_scaled+ 0.00001 )
X_test_transformed = pt.transform(X_test+0.000001)
pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_})


# In[ ]:





# In[ ]:




