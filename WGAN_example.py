#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:14:19 2022

@author: kesaprm
"""

from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot
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

## Get the gene p/a values of the corresponding genomes
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


in_shape = x_train.shape[1]
n_nodes = 2**13

# load data
def load_real_samples():
	# load dataset
	x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
	return x_train


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select samples
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    #x_input = randn(latent_dim * n_samples)
    # generate binary points in the latent space
    x_input = randint(2, size = latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


 
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
 
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# define the standalone critic model
def define_critic(in_shape=(in_shape,)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint: Constrain critic model weights to a limited range after each mini batch update [-0.01,0.01]
    const = ClipConstraint(0.01)
    # define model
    model = Sequential()
    ##Depth 1
    model.add(Dense(n_nodes, kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 2
    model.add(Dense(n_nodes, kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 3
    model.add(Dense(n_nodes, kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 3
    model.add(Dense(n_nodes, kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # scoring, linear activation
    model.add(Flatten())
    model.add(Dense(1))
    # compile model
    opt = RMSprop(lr=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    model.summary()
    return model

# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # define model
    model = Sequential()
    ##Depth 1
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 2
    model.add(Dense(n_nodes, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 3
    model.add(Dense(n_nodes, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    ##Depth 3
    model.add(Dense(n_nodes, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # output in_shapex1
    model.add(Dense(in_shape, activation='tanh', kernel_initializer=init))
    model.summary()
    return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = ones((n_samples, 1))
    X = (X + 1) / 2
    X = np.where(X > 0.5, 1, 0)
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, c_model, n_samples=10):
	# prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    #X = (X + 1) / 2
    print('> generate_fake_samples X' % X)
    # save the generated genome p/a
    filename1 = 'genGenome_%04d.csv' % (step+1)
    save_df = pd.DataFrame(X)
    save_df.to_csv(filename1)
    # save the generator model
    filename2 = 'Gmodel_%04d.h5' % (step+1)
    g_model.save(filename2)
     # save the critic model
    filename3 = 'Dmodel_%04d.h5' % (step+1)
    c_model.save(filename3)
    
    print('>Saved: %s and %s and %s' % (filename1, filename2, filename3))



# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()
    
# lists for keeping track of loss
c1_hist, c2_hist, g_hist = list(), list(), list()
# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % (n_steps/2) == 0:
			summarize_performance(i, g_model, latent_dim, c_model)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)

# size of the latent space
latent_dim =  2**11
# create the critic
critic = define_critic()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# load data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, critic, gan_model, dataset, latent_dim)

loss_df = pd.DataFrame(data = {"c1_hist":c1_hist,"c2_hist":c2_hist,"g_hist":g_hist })
loss_df.to_csv('WGAN_loss.csv')
