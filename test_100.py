#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:18:43 2022

@author: kesaprm
"""

## Import required libraries
import pandas as pd
import numpy as np



## Load the presence absence rtab data
pa_mat = pd.read_csv('gene_presence_absence.Rtab', sep='\t')

## Load the cloud genes
cloud_genes = pd.read_csv(r'cloud.txt', header=None)


## Pre processing step: Filter the cloud genes from the presene/absence data
df_new = pa_mat[pa_mat['Gene'].isin(cloud_genes[0]) == False]

## Get all the genome names from the dataframe header
genome_labels = df_new.columns[1:]

## Get the gene p/a values of the corresponding geneomes
genome_values = []
for k in range(0, len(genome_labels)):
    genome_values.append(df_new[genome_labels[k]].tolist())
    
## Getting ready to input the data to the model: conversion to an array   
genome_data = np.array(genome_values)


## Shuffle all the data before train/test split
from sklearn.utils import shuffle
all_data, all_labels = shuffle(genome_data, genome_labels, random_state=42)


## Split train:test = 80:20
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)


## Change the string labels to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

## One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

## Split train and validation 
x_val = x_train[:20]
partial_x_train = x_train[20:]

y_val = y_train_one_hot[:20]
partial_y_train = y_train_one_hot[20:]


## The generator  
import keras
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization


latent_dim = 100
height = x_train.shape[1]
width = x_train.shape[0]
kernel_size = 3

generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, input_shape=(x_train.shape[1],))(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((1,x_train.shape[1], 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Dense( 1, activation='tanh')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()

# def generator_model():
#     feature_extractor = Sequential()
#     feature_extractor.add(Dense(10, activation='relu',kernel_initializer='he_uniform',  input_shape=(x_train.shape[1],1)))
#     feature_extractor.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
#     feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
#     feature_extractor.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
#     feature_extractor.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
#     feature_extractor.add(Dense(10, kernel_initializer='he_uniform', activation='relu'))
#     feature_extractor.add(Flatten())
#     x = feature_extractor.output  
    
#     prediction_layer = Dense(3, activation = 'tanh')(x)
    
#     generator = Model(inputs=feature_extractor.input, outputs=prediction_layer)
#     generator.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     print(generator.summary()) 
#     return generator




## The discriminator
discriminator_input = layers.Input(shape=(height,1))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)  
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()


discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                     loss='binary_crossentropy')




## The adversarial network

# Sets discriminator weights to non-trainable (this will only apply to the gan model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


## GAN training
iterations = 10000
batch_size = 20
start = 0

for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size,
                                        latent_dim))
    
    generated_output = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_data = x_train[start: stop]
    combined_data = np.concatenate([generated_output, real_data])
    labels = np.concatenate([np.ones((batch_size, 1)),
                         np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_data, labels)
    
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    misleading_targets = np.zeros((batch_size, 1))
    
    a_loss = gan.train_on_batch(random_latent_vectors,
                            misleading_targets)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
        
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        
    print('discriminator loss:', d_loss)
    print('adversarial loss:', a_loss)
    

    
    
    