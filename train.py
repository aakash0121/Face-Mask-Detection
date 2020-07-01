import dataset_creator
import tensorflow as tf 
import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical

data = dataset_creator.create_data()

x=[]
y=[]
for features, labels in data:
    x.append(features)
    y.append(labels)

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)

x=np.array(x).reshape(-1,224,224,3)
x=tf.keras.utils.normalize(x,axis=1)

y = tf.keras.utils.to_categorical(y, num_classes=20)



