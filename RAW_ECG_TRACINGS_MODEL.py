#!/usr/bin/env python
# coding: utf-8

# ### IMPORTS

# In[1]:


import h5py
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import ecg_plot
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# In[2]:


dataset = h5py.File('ecg_tracings.hdf5', 'r')


# In[3]:


dataset.keys()


# In[4]:


tracings = dataset['tracings']


# ### RETRIEVING THE ECG TRACINGS IN AN ARRAY LIST

# In[5]:


a = np.asarray(tracings)


# In[6]:


a


# In[7]:


attributes = pd.read_csv('attributes.csv')


# In[8]:


attributes["sex"].replace({"M": 1, "F": 0}, inplace = True)


# ### GETTING THE ATTRIBUTES OF THE RESPECTIVE PATIENTS

# In[9]:


b = np.asarray(attributes)


# In[10]:


a.shape


# In[11]:


b.shape


# ### ADDING WEIGHTS BY NUMPY ZEROS TO EQUAL THE SIZE OF THE TWO ARRAYS

# In[12]:


z = np.zeros((827,4094))


# In[13]:


B=np.append(b, z, axis=1)


# In[14]:



B.shape


# In[15]:


B


# ### STACKING BOTH ARRAYS TO CREATE A TRAINABLE DATA

# In[16]:


X = np.dstack((a, B))


# In[17]:


X.shape


# In[18]:


y = pd.read_csv("dnn.csv")


# In[19]:


y.shape


# ### SIGNIFYING THE LABELS

# In[20]:


y = y.drop("Unnamed: 0",axis=1)
y


# ### PLOTTING A RANDOM ECG

# In[23]:


ecg = electrocardiogram()
frequency = 360
time_data = np.arange(ecg.size) / frequency
  
# plotting time and ecg model
plt.plot(time_data, ecg)
plt.xlabel("time in seconds")
plt.ylabel("ECG in milli Volts")


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(X,y, test_size=0.2,random_state=42, shuffle=True)


# In[25]:


model= keras.Sequential()
import tensorflow as tf


# #### CREATING AND APPLYING A CUSTOM DEEP NEURAL NETWORK CONSISTING OF CNN AND LSTM ON THE RAW ECG SIGNALS

# In[26]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[4096, 13]),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=False)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 200)])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
optimizer=optimizer,
metrics=["accuracy"])


# ### DISPLAYING THE ACCURACY AND LOSSES

# In[28]:


history = model.fit(x_train, Y_train, epochs=5, batch_size=256, validation_data=(x_test, Y_test))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


# In[29]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)


plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("water_quality_nn.pdf", dpi=100)
plt.show()

