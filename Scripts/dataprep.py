#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	For {regional, weekly, provincial}
	-Load cleaned mosquito data
	-Load geographic data
		-Perform KNN imputation
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(".../Dengue_modelling")

#Function to perform weighted KNN for every observation (even non-missing) with k = K, returns matrix of would-be imputed values
def weighted_KNN(K,wide,regions):
    #Which is the first column containing mosquito/count data, and which the last (needed later)
    col1 = 'mosquito_'+np.sort(regions)[0]
    col2 = 'mosquito_'+np.sort(regions)[regions.size - 1]
    col3 = 'count_'+np.sort(regions)[0]
    col4 = 'count_'+np.sort(regions)[regions.size - 1]
    KNN_values = wide.copy(deep=True)
    #For each region and each date, find k closest non-missing values and take the count-weighted average
    for r in np.sort(regions):
        for date in pd.unique(dataset['Date']):
                #Vectors of mosquito values and counts for that given date
                mosquitos = wide.loc[wide['Date']==date,col1:col2].to_numpy()
                counts = wide.loc[wide['Date']==date,col3:col4].to_numpy()
                #How many observations are non-missing and not the to-be imputed value itself?
                maxk = regions.size - 1 - np.isnan(mosquitos).sum() + np.isnan(wide.loc[wide['Date']==date,'mosquito_'+r])
                #From the distance matrix, obtain vector of distances to the region r
                distvec = distances[r].copy(deep=True).to_numpy()
                #Set distances for missing observations to inf. so they will not be used
                distvec[np.isnan(mosquitos)[0,:]] = float('inf')
                distvec[distvec==0] = float('inf')
                #Find the mosquito and count values of the k closest non-missing regions
                Kmosquitos = mosquitos[0,np.argsort(distvec)[0:min([maxk.iloc[0],K])]]
                Kcounts = counts[0,np.argsort(distvec)[0:min([maxk.iloc[0],K])]]
                #Obtain count-weighted K-mean of mosquito density
                KNN_values.loc[KNN_values['Date']==date,'mosquito_'+r] = (Kcounts * Kmosquitos).sum()/Kcounts.sum()
                print(date)
                print(r)
                print(KNN_values.loc[KNN_values['Date']==date,'mosquito_'+r])
    return KNN_values

#Function to write to csv a dataset in which the missing variables have been replaced by their KNN equivalents
def replaceKNN(KNNmat,wide,regions,dim):
    newdata = wide.copy(deep=True)
    count = 0
    for r in np.sort(regions):
        for date in pd.unique(dataset['Date']):
            if newdata.loc[newdata['Date']==date,'mosquito_'+r].isna().values:
                count += 1
                newdata.loc[newdata['Date']==date,'mosquito_'+r] = KNNmat.loc[KNNmat['Date']==date,'mosquito_'+r].values
    newdata.to_csv('Data/KNN_'+dim+'.csv')
    print(count)
    return newdata

#For each dataset, load the data, the distance matrix, set to wide format rather than long, and output dataset
for dim in ['region','weekly','province']:
    k = 0
    filename = "Data/Dataprep/merged_" + dim +".csv"
    dataset = pd.read_csv(filename)
    print(dataset.columns)
    distances = pd.read_csv(('Data/Dataprep/distance_'+dim+'.csv')).rename(columns = {'Unnamed: 0':'Region','MI MAROPA': 'IV-B'},index = {'MI MAROPA':'IV-B'})

    wide = pd.DataFrame(data={'Date': pd.unique(dataset['Date'])})
    dataset['Province'] = dataset['Province'].replace('MI MAROPA','IV-B')
    regions = np.sort(pd.unique(dataset['Province']))
    #Reshape to wide format: Dengue and Mosquito column for each region
    for r in regions:
        wide['dengue_'+r] = dataset[dataset['Province']==r]['Dengue'].to_numpy()
        wide['mosquito_'+r] = dataset[dataset['Province']==r]['value'].to_numpy()
        wide['count_'+r] = dataset[dataset['Province']==r]['Counts'].to_numpy()
    wide = wide.reindex(sorted(wide.columns), axis=1)
    distances = distances.set_index('Region')
    KNN_result = weighted_KNN(3,wide,regions)
    replaceKNN(KNN_result,wide,regions,dim)
