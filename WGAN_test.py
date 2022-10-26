#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:26:54 2022

@author: kesaprm
"""


## Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

## Real genome labels
true_labels_for_training = ['True']* len(genome_labels)

## Shuffle all the data before train/test split
from sklearn.utils import shuffle
all_data, all_labels = shuffle(genome_data, true_labels_for_training, random_state=42)


## Split train:test = 80:20
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)


## Change the string labels to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)

y_train_encoded = np.asarray(y_train_encoded).astype('float32')
y_test_encoded = np.asarray(y_test_encoded).astype('float32')


# from keras import models
# from keras import layers


# ### Discriminator
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop',
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'])


# history = model.fit(x_train,
#                     y_train_encoded,
#                     epochs=20,
#                     batch_size=20)

    
# import matplotlib.pyplot as plt
# history_dict = history.history
# loss_values = history_dict['loss']
# epochs = range(0, 20)
# plt.plot(epochs, loss_values, 'b-', label='Training loss')
# acc_values = history_dict['accuracy'] 
# plt.plot(epochs, acc_values, 'g-', label='Training acc')

# plt.title('Training')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# results = model.evaluate(x_test, y_test_encoded)



## Using functional API -discriminator
import keras
from keras import Input, layers
from keras.models import Model
from keras import backend

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

## Weight clipping
Constraint = 0.01
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
...
# define the constraint
const = ClipConstraint(Constraint)

discriminator_input = Input(shape=(x_train.shape[1],))
x = layers.Dense(512, kernel_constraint=const)(discriminator_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dense(512, kernel_constraint=const)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)

## #WGAN uses linear activation
x = layers.Dense(1, activation='linear')(x)

discriminator = Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.00005)

discriminator.compile(optimizer=discriminator_optimizer,
                     loss= wasserstein_loss)


## Using functional API -generator
latent_dim = x_train.shape[1]

generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(512)(generator_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dense(x_train.shape[1], activation='tanh')(x)
generator = Model(generator_input, x)
generator.summary()



## GAN
from keras.layers import BatchNormalization

for layer in discriminator.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,)) 
gan_output = discriminator(generator(gan_input)) 
gan = Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.00005) 
gan.compile(optimizer=gan_optimizer, loss= wasserstein_loss)

## Training

iterations = 100
batch_size = 20

dis_loss = []
gen_loss = []
start = 0
n_critic = 5
for step in range(iterations):
    
    for _ in range(n_critic):
        random_latent_vectors = np.random.normal(size=(batch_size,
                                            latent_dim))
        generated_matrix = generator.predict(random_latent_vectors)
        
        stop = start + batch_size
        real_matrix = x_train[start: stop]
        
        combined_data = np.concatenate([generated_matrix, real_matrix])
        
        ## change in wgan labels : -1  for real data and 1 for fake data 
        labels = np.concatenate([-np.ones((batch_size, 1)),
                             np.ones((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        
        d_loss = discriminator.train_on_batch(combined_data, labels)
    
    
    
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    misleading_targets = np.ones((batch_size, 1))
    a_loss = gan.train_on_batch(random_latent_vectors,
                            misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
        
    if step % 10 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        dis_loss.append(d_loss)
        gen_loss.append(a_loss)
        


plt.plot(range(0, len(dis_loss)*100,100), dis_loss, label = 'Discriminator Loss', color = 'b')    
plt.plot(range(0, len(gen_loss)*100,100), gen_loss, label = 'Generator Loss', color = 'g')  
#plt.legend()

plt.xlabel('Iterations')
plt.ylabel('Generator Loss')
plt.show()
