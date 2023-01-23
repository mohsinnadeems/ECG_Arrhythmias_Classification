#!/usr/bin/env python
# coding: utf-8

# ### IMPORTS

# In[1]:


from __future__ import division
from matplotlib import pyplot as plt
import scipy.io as spio
import numpy as np
import statistics
from scipy.stats import kurtosis
from scipy.stats import skew,tstd
import sys
import cProfile
from functools import partial
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder


# In[2]:


import os
import keras
from keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout,Activation
from tensorflow.keras import optimizers
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


# ### RETRIEVING THE DATA FROM THE DATA FOLDER

# In[3]:


directory = "Downloads/WFDBRecords"
for filename in os.listdir(directory):
    if filename.endswith(".dat") or filename.endswith(".hea"): 
         print(os.path.join(directory, filename))
    else:
        continue


# ### CREATING EMPTY LISTS AND APPENDING THE ECG SIGNALS AND HEADER FILES

# In[4]:


# import required module
import os
 
# assign directory
directory = 'Downloads/WFDBRecords'
mat_files = []
hea_files = []


for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".mat"):
            mat_files.append(os.path.join(root, filename))
        else:
            hea_files.append(os.path.join(root, filename))


# In[5]:


mat_files


# In[6]:


hea_files


# ### ENUMERATING RANDOM 5000 PATIENTS ECG TRACINGS

# In[7]:


data = []
for num, i in enumerate (mat_files[0:5000]):
    df = spio.loadmat(i)
    data.append(df.get('val'))


# In[8]:


data


# In[9]:


len(data)


# ### EXTRACTING THE FEATURES

# In[10]:


def absolute_diff(x):
    length_int = len(x)
    diff_nni = np.diff(x)
    nni_50 = np.sum((np.abs(diff_nni) > 50),axis=2)
    return 100 * nni_50 / length_int


# In[11]:


fea1 = absolute_diff(data)


# In[12]:


fea1


# In[13]:


def differenciate(list):
    diff_nni = np.diff(list)
    return np.sum((np.abs(diff_nni) > 50),axis=2)


# In[14]:


fea2 = differenciate(data)


# In[15]:


fea2


# In[16]:


def kurt(x):
    return kurtosis(x,axis=2)


# In[17]:


fea3 = kurt(data)


# In[18]:


fea3


# In[19]:


def skewness(x):
    return skew(x,axis=2)


# In[20]:


fea4 = skewness(data)


# In[21]:


fea4


# In[22]:


def standard_deviation(x):
    return tstd(x,axis=2)


# In[23]:


fea5 = standard_deviation(data)


# In[24]:


fea5


# ### CONVERTING ALL THE EXTRACTED FEATURES TO A DATAFRAME

# In[25]:


df_1 = pd.DataFrame(fea1)
df_2 = pd.DataFrame(fea2, columns=[x for x in range(12,24)])
df_3 = pd.DataFrame(fea3, columns=[x for x in range(24,36)])
df_4 = pd.DataFrame(fea4, columns=[x for x in range(36,48)])
df_5 = pd.DataFrame(fea5, columns=[x for x in range(48,60)])


# ### RETRIEVING THE AGE, SEX AND THE LABELS STORED IN THE HEADER FILE

# In[26]:


age = []
sex = []
dx = []

for i in hea_files:
    with open(i, 'r') as the_file:
        all_data = [line.strip() for line in the_file.readlines()]
        data = all_data[13:16]
        age.append(data[0].lstrip('#Age: '))
        sex.append(data[1].lstrip('#Sex: '))
        dx.append(data[2].lstrip('#Dx: '))


# In[27]:


df_cl = pd.DataFrame({'Age': age, 'Sex': sex, 'Dx':dx})
df_cl= df_cl[0:5000]


# ### CONCATINATING ALL THE DATAFRAMES

# In[28]:


df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_cl], axis=1)


# In[29]:


df


# In[30]:


df = df[df["Dx"].str.contains("JS") == False]


# In[31]:


df


# ### ENCODING THE MALE AND FEMALES BY 0 AND 1

# In[32]:


df["Sex"].replace({"Male": 1, "Female": 0},inplace=True)


# In[33]:


df['Sex'].unique()


# In[34]:


df = df[df.Sex != 'Dx: 427084000']


# In[35]:


df


# In[36]:


df.reset_index(inplace=True)


# In[37]:


df.drop('index',axis=1,inplace=True)


# In[38]:


df


# ### PLOTTING THE GRAPH OF AGE VS SEX SUFFERING FROM CARDIAC DISEASES

# In[39]:


demographic = df[['Age', 'Sex']]

demographic['Age'] = demographic['Age'].astype(int)

def ageband(val):
    if val >= 61:
        return '61-above'
    elif val >= 41:
        return '41-60'
    elif val >= 21:
        return '21-40'
    else:
        return '00-20'
demographic['Age Band'] = 0    
for index, row in enumerate(demographic['Age']):
    demographic['Age Band'][index] = ageband(row)

    
demographic["Sex"].replace({1: "Male", 0: "Female"},inplace=True)
    
demographic

# Computing Age Pyramid
import seaborn as sns

male = list(0 for i in range(4))

for i in demographic[(demographic['Sex'] == 'Male')]['Age']:
    if (i >= 61):
        male[0] += 1
    
    elif (i <= 60) & (i >= 41):
        male[1] += 1
    
    elif (i <= 40) & (i >= 21):
        male[2] += 1
        
    elif (i <= 20) & (i >= 0):
        male[3] += 1   
        
female = list(0 for i in range(4))

for i in demographic[(demographic['Sex'] == 'Female')]['Age']:
    if (i >= 61):
        female[0] += 1
    
    elif (i <= 60) & (i >= 41):
        female[1] += 1
    
    elif (i <= 40) & (i >= 21):
        female[2] += 1
        
    elif (i <= 20) & (i >= 0):
        female[3] += 1

male = np.array(male) * -1

male = male.tolist()

age_p = pd.DataFrame({'Age': ['61-above', '41-60', '21-40', '00-20'],
                      'Male': male,
                      'Female': female})

AgeClass = ['61-above', '41-60', '21-40', '00-20']


age_pyramid = sns.barplot(x='Male', y='Age', data=age_p, order=AgeClass, color=('blue'), label='Male')
age_pyramid = sns.barplot(x='Female', y='Age', data=age_p, order=AgeClass, color=('pink'), label='Female')
age_pyramid.set(xlabel="Population", ylabel="Age-Group", title = "Population Pyramid")
age_pyramid.legend()
plt.title('Age Pyramid')


# In[40]:


X = df.drop('Dx',axis=1)


# In[41]:


y_unfilled = df['Dx']


# In[42]:


y_unfilled


# In[43]:


y_new = y_unfilled.values


# In[44]:


new_list=[]
for i in range(4950):
    new_list.append(y_new[i].split(','))


# In[45]:


new_list


# ### MAKING EVERY LABEL OF EQUAL SIZE

# In[46]:


l = len(new_list)
for i in range(l):
    if len(new_list[i]) < 6:
        new_list[i].extend(None for _ in range(6 - len(new_list[i]))) 
    


# In[47]:


len(new_list)


# In[48]:


y_draft =pd.DataFrame(new_list)


# In[49]:


y_draft


# In[50]:


y1 = pd.get_dummies(y_draft[0])
y2 = pd.get_dummies(y_draft[1])
y3 = pd.get_dummies(y_draft[2])
y4 = pd.get_dummies(y_draft[3])
y5 = pd.get_dummies(y_draft[4])
y6 = pd.get_dummies(y_draft[5])


# In[51]:


y_ = pd.concat([y1,y2,y3,y4,y5,y6],axis=1)


# In[52]:


y_


# ### MAKING THE COLUMNS OF EVERY PROBABLE OUTCOME

# In[53]:


y =y_.groupby(y_.columns, axis=1).sum()


# In[54]:



y


# In[55]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### NORMALIZING AND SPLITING THE DATA
# 

# In[56]:


std = StandardScaler()


# In[57]:


X = std.fit_transform(X)


# In[58]:


X


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[60]:


X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


# ### CREATING THE DEEP NEURAL NETWORK

# In[61]:


def Model(input_dim, activation, num_class):
    model = Sequential()
    

    model.add(Dense(2048, input_dim = input_dim))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(64))    
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(num_class))    
    model.add(Activation('linear'))
   
    model.compile(loss='Huber',
                  optimizer=optimizers.Adam(lr = 0.001),
                  metrics=['MeanAbsoluteError','Accuracy']
                 )
    return model


# In[62]:


Model(62, 'Softmax',47)


# In[63]:


input_dim = X_train.shape[1]
activation = 'relu'
classes = 1

learning_rate=0.5
model = Model(input_dim=input_dim, activation=activation, num_class=classes)
model.summary()


# ### PLOTTING THE GRAPH ON 50 EPOCHS AND MEASURING THE ACCURACY|

# In[64]:


history = model.fit(X_train, y_train, epochs=50, batch_size=132, validation_data=(X_test, y_test))
plt.plot(history.history['Accuracy'], label='train_accuracy')
plt.plot(history.history['val_Accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[65]:


losses = pd.DataFrame(model.history.history)


# ### CREATING THE LOSS GRAPH

# In[66]:


plt.figure(figsize=(15,10))
losses.plot()

