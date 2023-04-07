# data-science
#files for downloading excel and csv for further data analysis
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
model= keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),
    keras.layers.Dense(100, activation= 'sigmoid'),
    keras.layers.Dense(10, activation= 'sigmoid')
])
tb_callback= tf.keras.callbacks.TensorBoard(log_dir= "keras/", histogram_freq=1)
model.compile(
    optimizer= 'SGD',
    loss= 'sparse_categorical_crossentropy',
    metrics= ['accuracy']
)
model.fit(x_train,y_train,epochs=5, callbacks= [tb_callback]) 
