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
gen_mat = pd.read_csv('genGenome_0001.csv')
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
plt.savefig('Real_pers_counts.png')

plt.hist(pers_count, weights=np.ones(len(pers_count)) / len(pers_count))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig('Generated_pers_counts.png')

##to plot a seaborn plot
gen_new_per.to_csv('gen_seaborn.csv')


### Clustered heatmap
import seaborn as sns

import sys
sys.setrecursionlimit(100000)


gen_seaborn = pd.read_csv('genGenome_0015.csv')
# drop the first column
gen_seaborn = gen_seaborn.iloc[: , 1:]
gen_seaborn = (gen_seaborn + 1) / 2
gen_seaborn = np.where(gen_seaborn > 0.5, 1, 0)

# gen_new_per.set_index('Gene', inplace=True)
# gen_seaborn.set_index('Gene', inplace=True)

# x_axis_labels = np.ones(32).tolist()#['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10'] # labels for x-axis

# sns.clustermap(gen_new_per,  cmap="viridis", xticklabels=x_axis_labels)
gen_seaborn = gen_seaborn.T

sns.clustermap(gen_seaborn,  cmap="viridis")

plt.show()


## check the number of genes detected by the generated genomes as present
countPresent_genes = []

for r in range(0, len(gen_seaborn)):
    g1 = gen_seaborn[r]
    # Get the count of genes present (1s) in a genomes
    count = (g1 == 1).sum()
    countPresent_genes.append(count)
    

df_gen_gene_counts = pd.DataFrame(countPresent_genes, columns = ['Gene_Counts'])

df_gen_gene_counts['percent_present'] = round(df_gen_gene_counts.Gene_Counts/gen_seaborn.shape[1]*100)

plt.hist(df_gen_gene_counts['percent_present'] )

plt.box(df_gen_gene_counts['percent_present'])
#sns.clustermap(x_train,  cmap="viridis")


"""==========Discriminator validation============"""
# define the standalone critic model
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
opt = RMSprop(learning_rate=0.00005)
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

y_label = -np.ones((len(x_test), 1))

test_loss  = D.evaluate(x_test, y_label.astype(float))

#y_predict = D.predict_classes(x_test)

predictions = (D.predict(x_test) > 0.5).astype("int32")

unique, counts = np.unique(predictions, return_counts=True)

np.savez('mat_test_pred.npz', name1=unique, name2=counts)


### Discriminator validation with fake data generated by random 

x_fake = np.zeros(len(gene_labels))
x_fake2 = np.zeros(len(gene_labels))
x_fake3 = np.zeros(len(gene_labels))

x_fake3[234] = 1.

x_f = np.array([x_fake.tolist(),x_fake2.tolist(),x_fake3.tolist(),randint(2, size = len(gene_labels)).tolist()])

x_f =  x_f.reshape(len(x_f), len(gene_labels))
y_f = np.ones((len(x_f), 1))

val_loss  = D.evaluate(x_f,  y_f.astype(float))

print('d_loss with fake arrays:', val_loss)

predictionsF = (D.predict(x_f) > 0.5).astype("int32")

uniqueF, countsF = np.unique(predictionsF, return_counts=True)

np.savez('mat_test_pred_fake.npz', name1=uniqueF, name2=countsF)

#plt.hist(listD, weights=np.ones(len(listD)) / len(listD))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))




