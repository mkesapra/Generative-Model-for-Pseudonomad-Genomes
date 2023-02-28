#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:30:41 2022

@author: kesaprm
"""

from pathlib import Path
import pandas as pd
import numpy as np
import glob
import os


path = r'/Users/kesaprm/Pseudomonads/gene_position/'


# get the files from the path
files = Path(path).glob('*.tsv')  # .rglob to get subdirectories
#files = glob.glob(os.path.join(path , "*.tsv"))


# dfs = list()
# for f in files:
#     data =  pd.read_table(f)#pd.read_csv(f, sep='\t')
#     # .stem is method for pathlib objects to get the filename w/o the extension
#     data['fileName'] = f.stem
#     data['isPair'] = 0
#     dfs.append(data)

def getEachGenome(gen_A):
    data = pd.read_table(gen_A)
    data['fileName'] = gen_A.stem
    data['isPair'] = 0
    return data


dfs = list(map(getEachGenome, files))


def findPairs(df_A):
    df_A['contig_shift'] = df_A['contig'].shift(1)
    df_A['strand_shift'] = df_A['strand'].shift(1)
    
    ## update the isPair column to 1 if the contig and strand are same
    df_A.loc[(df_A.contig_shift == df_A.contig) & (df_A.strand_shift == df_A.strand), 'isPair'] = 1
    
    ## find pairs
    familyPairs = []
    for i in range(1, len(df_A)):
        if (df_A.at[i,'isPair'] == 1):
            familyPairs.append((df_A.at[i,'family'],df_A.at[i-1,'family']))
  
    return familyPairs


## create a pairs list for all the input genomes
outputlist = list(map(findPairs, dfs))

## find core families in all the list of famililes
coreFamilies = list(set.intersection(*map(set, outputlist)))

print("Core Families====> ",coreFamilies)


## unique Families
def getUniqueFamilies(list_A):
    return list(set(list_A) - set(coreFamilies))

## get uniqueGenomeNames
def uniqueGenomeNames(df_A):
    return df_A.at[0,'fileName']

## create a unique families list for all the genomes removing the core families from them
uniqueFamilies = list(map(getUniqueFamilies,outputlist))
uniqueGenomeNames = list(map(uniqueGenomeNames,dfs))

## unique families dataframe
df_uniqueFamilies = (pd.DataFrame(uniqueFamilies,uniqueGenomeNames)).T

df1 = df_uniqueFamilies.stack().reset_index(-1).iloc[:, ::-1]
df1.columns = ['familyPairs', 'genomeName']


familyPairs_pa_mat = pd.crosstab(df1['familyPairs'], df1['genomeName'])

familyPairs_pa_mat['new1'] = (familyPairs_pa_mat.sum(axis=1)/len(dfs)).mul(100).astype(int)

familyPairs_pa_mat2 = familyPairs_pa_mat[familyPairs_pa_mat['new1']> 10]

familyPairs_pa_mat2.to_csv('familyPairs_pa_mat.csv')



import seaborn as sns

familyPairs_pa_mat100 = pd.read_csv('family_100.csv')

sns.heatmap(familyPairs_pa_mat100, cmap="viridis",  yticklabels=False,  xticklabels=False)


# ## create a family presene absence list between all the genomes
# from functools import reduce
# rows = reduce(lambda count, l: count + len(l), uniqueFamilies, 0)
# cols = len(uniqueFamilies)

# familyPairs_pa_mat = np.ones((rows,cols))

# ## check if the similar family pairs are present in other genomes, if not make the pa-value 0




