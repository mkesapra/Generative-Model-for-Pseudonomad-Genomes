#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:14:19 2022

@author: kesaprm
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from numpy import ones
from numpy.random import randint
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
import os 

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
# size of the latent space
latent_dim =  2**6
batch_size=32

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

# define the standalone critic model
def define_critic(in_shape=(in_shape,)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint: Constrain critic model weights to a limited range after each mini batch update [-0.01,0.01]
    # define model
    model = Sequential()
    ##Depth 1
    model.add(Dense(n_nodes, kernel_initializer=init, input_shape=in_shape))
    model.add(BatchNormalization())
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
    # scoring, linear activation
    model.add(Flatten())
    model.add(Dense(1))
    # compile model
    #opt = RMSprop(lr=0.00005)
    #model.compile(loss=wasserstein_loss, optimizer=opt)
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


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size,in_shape], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}



class SaveModel(keras.callbacks.Callback):                       
    def __init__(self, save_path: str):
          self.save_path = save_path

          
    def on_epoch_end(self, epoch: int, logs: dict):
        #if (epoch+1) % 50 == 0:     
        filename1 = 'Gmodel_%04d.h5' % (epoch+1)
        filename2 = 'Dmodel_%04d.h5' % (epoch+1)
        filename3 = 'genGenome_%04d.csv' % (epoch+1)
        # Save the generator model                  
        self.model.generator.save(filename1)
        # Save the critic model                  
        self.model.discriminator.save(filename2)
        
        # prepare fake examples
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = self.model.generator(random_latent_vectors, training=False)
        # scale from [-1,1] to [0,1]
        #X = (X + 1) / 2
        # save the generated genome p/a
        X = tf.keras.backend.get_value(generated_images)
        df2 = pd.DataFrame(X)
        df2.to_csv(filename3)
        
        
    
  
# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Set the number of epochs for trainining.
epochs = 1


# Get the wgan model
wgan = WGAN(
    discriminator=define_critic(),
    generator=define_generator(latent_dim),
    latent_dim=latent_dim,
    discriminator_extra_steps=3,
)

# Compile the wgan model
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Start training
wgan.fit(np.float32(x_train), batch_size=32, epochs=epochs, callbacks = SaveModel(save_path = '/output_WGP/'))

