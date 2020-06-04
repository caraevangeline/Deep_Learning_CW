%tensorflow_version 1.x
%matplotlib inline

#-------------------Import required libraries----------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from keras.layers import Flatten, Activation, Conv2D, MaxPool2D, AvgPool2D, Dense, Dropout, BatchNormalization, Input, MaxPooling2D, Flatten, Activation, Conv2D, AvgPool2D, Dense, Dropout, concatenate, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.models import Sequential
import keras.backend as K
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import model_from_json, Model
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10

np.random.seed(451)

#-----------------------Load the dataset-----------------#
#---------------------(MNIST)----------------------------#
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
x_train = mnist.train.images  # Returns np.array
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
x_test = mnist.test.images  # Returns np.array
y_test = np.asarray(mnist.test.labels, dtype=np.int32)
#------------------(CIFAR 10)------------------------#
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#----------------Normalize the dataset-----------------#
x_train = x_train / 255.0
x_test = x_test / 255.0

#---------------Reshape the dataset-----------------#
#--------------(MNIST)------------------------------#
x_train_gray = x_train.reshape(-1,28,28,1)
x_test_gray = x_test.reshape(-1,28,28,1)
#--------------(CIFAR 10)--------------------------#
#x_train_gray = np.dot(x_train[:,:,:,:3], [0.299, 0.587, 0.114])
#x_test_gray = np.dot(x_test[:,:,:,:3], [0.299, 0.587, 0.114])
#x_train_gray = x_train_gray.reshape(-1,32,32,1)
#x_test_gray = x_test_gray.reshape(-1,32,32,1)

#---------------Convert to catergorical value---------#
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#--------------Display one sample train image---------#
plt.imshow(x_train_gray[1,:,:,0], cmap='gray')
plt.show()


np.random.seed(451)


#-------------Define the functions and modules present in GoogleNet structure-----#
def build_tower(input_layer, features_nr, shape, tower_nr, 
                dropout=False, normalization=False, regularization="l2", dropout_ratio=0.25):
    #3x3 kernel tower
    tower = Conv2D(features_nr, (1,1), padding='same', activation='relu', 
                     kernel_regularizer=regularization, name='tower_%d_%dx%da'%(tower_nr, shape[0], shape[1]))(input_layer)
    tower = Conv2D(features_nr*2, shape, padding='same', activation='relu',
                     kernel_regularizer=regularization, name='tower_%d_%dx%db'%(tower_nr, shape[0], shape[1]))(tower)
    #condidional dropout/normalization
    if dropout:
        tower = Dropout(dropout_ratio, name='tower_%d_%dx%ddrop'%(tower_nr, shape[0], shape[1]))(tower)
    if normalization:
        tower = BatchNormalization(name='tower_%d_%dx%dnorm'%(tower_nr, shape[0], shape[1]))(tower)
        
    return tower

def build_simple_tower(input_layer, features_nr, shape, tower_nr, 
                dropout=False, normalization=False, regularization="l2", dropout_ratio=0.25):
    #3x3 kernel tower
    tower = Conv2D(features_nr, shape, padding='same', activation='relu',
                     kernel_regularizer=regularization, 
                   name='tower_simple_%d_%dx%db'%(tower_nr, shape[0], shape[1]))(input_layer)
    #condidional dropout/normalization
    if dropout:
        tower = Dropout(dropout_ratio, name='tower_%d_%dx%ddrop'%(tower_nr, shape[0], shape[1]))(tower)
    if normalization:
        tower = BatchNormalization(name='tower_%d_%dx%dnorm'%(tower_nr, shape[0], shape[1]))(tower)
        
    return tower

def build_tower_subsample(input_layer, features_nr, shape, tower_nr, 
                          dropout=False, normalization=False, regularization='l2', dropout_ratio=0.25):
    tower = build_tower(input_layer, features_nr, shape, tower_nr, 
                        dropout, normalization, regularization, dropout_ratio)
    pool = MaxPooling2D((2,2), padding='same', name='tower_%d_2x2subsample'%(tower_nr))(tower)

    return pool

def build_simple_tower_subsample(input_layer, features_nr, shape, tower_nr, 
                          dropout=False, normalization=False, regularization='l2', dropout_ratio=0.25):
    tower = build_simple_tower(input_layer, features_nr, shape, tower_nr, 
                        dropout, normalization, regularization, dropout_ratio)
    pool = MaxPooling2D((2,2), padding='same', name='tower_%d_2x2subsample'%(tower_nr))(tower)

    return pool

def build_dense(input_layer, neurons_nr, dense_nr, 
                dropout=False, normalization=False, regularization='l2', dropout_ratio=0.5):
    dense = Dense(neurons_nr, kernel_regularizer=regularization, 
                  name='dense_%d_%d'%(dense_nr, neurons_nr))(input_layer)
    
    if dropout:
        dense = Dropout(dropout_ratio, name='dense_%d_%ddrop'%(dense_nr, neurons_nr))(dense)
    if normalization:
        dense = BatchNormalization(name='dense_%d_%dnorm'%(dense_nr, neurons_nr))(dense)
    
    return dense

def build_inception_module(input_layer, features_nr, module_nr, 
                           dropout=False, normalization=False, regularization='l2', dropout_ratio=0.2):  
    #feature_nr is an array we'll use to build our layers
    #data is in the form: [1x1, 3x3 reduce, 3x3, 5x5 reduce, 5x5, pool proj]
  
    inception_1x1 = Conv2D(features_nr[0],1,1,border_mode='same',activation='relu',name='inception_%d_/1x1'%(module_nr),W_regularizer=l2(0.0002))(input_layer)
    
    inception_3x3_reduce = Conv2D(features_nr[1],1,1,border_mode='same',activation='relu',name='inception_%d_/3x3_reduce'%(module_nr),W_regularizer=l2(0.0002))(input_layer)
    
    inception_3x3 = Conv2D(features_nr[2],3,3,border_mode='same',activation='relu',name='inception_%d_/3x3'%(module_nr),W_regularizer=l2(0.0002))(inception_3x3_reduce)
    
    inception_5x5_reduce = Conv2D(features_nr[3],1,1,border_mode='same',activation='relu',name='inception_%d_/5x5_reduce'%(module_nr),W_regularizer=l2(0.0002))(input_layer)
    
    inception_5x5 = Conv2D(features_nr[4],5,5,border_mode='same',activation='relu',name='inception_%d_/5x5'%(module_nr),W_regularizer=l2(0.0002))(inception_5x5_reduce)
    
    inception_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_%d_/pool'%(module_nr))(input_layer)
    
    inception_pool_proj = Conv2D(features_nr[5],1,1,border_mode='same',activation='relu',name='inception_%d_/pool_proj'%(module_nr),W_regularizer=l2(0.0002))(inception_pool)
    
    inception_output = concatenate([inception_1x1,inception_3x3,inception_5x5,inception_pool_proj],axis=3,name='inception_%d_/output'%(module_nr))

    if dropout:
        inception_output = Dropout(dropout_ratio, name='inception_%d_/output_drop'%(module_nr))(inception_output)
    if normalization:
        inception_output = BatchNormalization(name='inception_%d_/output_norm'%(module_nr))(inception_output)

    pooled = MaxPooling2D((2,2), padding='same', name='inception_%d_2x2subsample'%(module_nr))(inception_output)
    
    return pooled

i='mnist-nrcrt7-'+datetime.datetime.now().strftime("%I:%M%p_%B-%d-%Y")

K.clear_session()

!mkdir -p models
!mkdir -p logs

a = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')#will stop the model if val_loss does not improve for 2 consecutive epochs
b = ModelCheckpoint(monitor='val_loss', filepath='./models/'+str(i)+'.hdf5', verbose=1, save_best_only=True)#save model weights after each epoch if val_loss improves
c = TensorBoard(log_dir='./logs/'+str(i),
                write_grads=True,
                write_graph=True,
                write_images=True,
                batch_size=128)#saves a log file for tensorboard; remember to save different runs to different subdirectories

#we'll use this instead of decay
d = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[a,b,c,d]

#------------model definition-------------------
use_norm = True
lrate = 0.001
input_img = Input(shape = (28, 28, 1), name='input') #....(32,32,3) for Cifar 10 dataset..#

inception_1 = build_inception_module(input_img, [64,96,128,16,32,32], 1, False, use_norm)
inception_2 = build_inception_module(inception_1, [128,128,192,32,96,64], 2, False, use_norm)
inception_3 = build_inception_module(inception_2, [192,96,208,16,48,64], 3, False, use_norm)
inception_4 = build_inception_module(inception_3, [160, 112, 224, 24, 64, 64], 4, False, use_norm)
flat_pool = AveragePooling2D(pool_size=(2, 2), padding='valid')(inception_4)
flat = Flatten()(flat_pool)
dense_5 = build_dense(flat, 128, 1, True, use_norm)
dense_6 = build_dense(dense_5, 64, 2, True, use_norm)
out = Dense(10, activation='softmax')(dense_6)
model = Model(inputs = input_img, outputs = out)

#----------------Compile the model and save-------------------------------#
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lrate),
              metrics=['accuracy'])
model.summary()
model_json = model.to_json()
with open("./models/"+str(i)+".json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to" + "../models/"+str(i)+".json")


#----------------Train the model and evaluate on test data--------------------------------------------#
with tf.device('/gpu:0'):
  model.fit(x_train_gray, y_train_cat, batch_size=128, epochs=100, validation_split=0.2,verbose=1,callbacks=callbacks)  # starts training #..x_train for Cifar10
result = model.evaluate(x_test_gray, y_test_cat)  #....x_test for Cifar 10
print("Accuracy on test set: ",result[1]*100,"%")

#----------------Evaluate the result(Display test accuracy and loss)-------------------#
model.load_weights('./models/mnist-nrcrt7-06:48PM_May-04-2020.hdf5')
result = model.evaluate(x_test_gray, y_test_cat)  #...x_test for cifar 10
print(result)

#---------------Predict and output predicted and actual class of one test sample-------------#
predict = model.predict(x_test_gray)  #...x_test for cifar 10
m = max(predict[0])
index = [i for i,j in enumerate(predict[0]) if j == m]
print("The value of the prediction of test sample with index 0 is :")
print(index)
print("The actual class of test sample with index 0 is: ")
plt.imshow(x_test_gray[0,:,:,0], cmap='gray') #...x_test for cifar 10
plt.show()





