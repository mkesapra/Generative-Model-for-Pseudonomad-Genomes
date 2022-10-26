#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:57:54 2022

@author: kesaprm
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import random 
from matplotlib.ticker import PercentFormatter
from keras.models import load_model, Sequential
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

## Load the presence absence rtab data
pa_mat = pd.read_csv('gene_presence_absence.Rtab', sep='\t')

## Load the cloud and shell genes
cloud_genes = pd.read_csv(r'cloud.txt', header=None)
shell_genes = pd.read_csv(r'shell.txt', header=None)

## Load the persistent genes
persistent_genes = pd.read_csv(r'persistent.txt', header=None)

## Pre processing step: Filter the cloud genes from the presene/absence data
df_new = pa_mat[pa_mat['Gene'].isin(cloud_genes[0]) == False]

df_per = pa_mat[pa_mat['Gene'].isin(persistent_genes[0]) == True]

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

"""======Generator validation=========="""
## Map the genome-gene binary matrix to the labels
gene_labels = df_new.Gene.tolist()

## Load the generated genome data
gen_mat = pd.read_csv('genGenome_0010.csv')
genome_len = len(gen_mat)

# Transpose the matrix
gen_mat = gen_mat.T

#remove the first row which was addded additionally from the csv
gen_mat = gen_mat.iloc[1: , :]

# Add the gene labels as first column to the generated genome data
gen_mat.insert(0, 'Gene', gene_labels)


# To verify if the persistent genes are present
gen_new_per = gen_mat[gen_mat['Gene'].isin(persistent_genes[0]) == True]

## To check how many persistent genes were correctly present
pers_count = []

for j in range(0, genome_len):
    pers_genes = gen_new_per[j].value_counts()[1]
    pers_count.append(pers_genes)
    

## cross validate with actual genome persistent genes
real_pers_count = []

real_genome_len = pa_mat.shape[1]
for j in range(1, real_genome_len):
    pers_genes = df_per.iloc[:,j].value_counts()[1]
    real_pers_count.append(pers_genes)
    
plt.hist(real_pers_count, weights=np.ones(len(real_pers_count)) / len(real_pers_count))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

plt.hist(pers_count, weights=np.ones(len(pers_count)) / len(pers_count))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

"""==========Discriminator validation============"""
# define the standalone critic model
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
opt = RMSprop(lr=0.00005)
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


D = load_model('Dmodel_0010.h5', compile=False,custom_objects={'ClipConstraint':ClipConstraint})
D.summary()
D.compile(loss=wasserstein_loss, optimizer=opt)

y_label = -np.ones((20, 1))

test_loss  = D.evaluate(x_test, y_label.astype(np.float))

y_predict = D.predict_classes(x_test)




