# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:16:20 2020

@author: sarak
"""
import os
import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from linearmodels.panel import PooledOLS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statsmodels.api as sm
os.chdir("/Users/sarak/Erasmus University Rotterdam/Daan Wassenberg - 510Drive/Dengue_modelling")

#load prepared province data
#Here the weekly data is also loaded. Methods are applied to the province level,
#but can also be directly(!) applied to the weekly data set
dfProvince  = pd.read_csv('Data/Research Question 2 Final/province_data_prepared.csv')

dfProvince['Date'] = pd.to_datetime(dfProvince['Date'])
dfProvince = dfProvince.set_index(['Province', 'Date'])
dfProvince['Provincecol']= dfProvince.index.get_level_values('Province')

#all methods which follow can be applied to the data set on weekly level:
dfWeekly = pd.read_csv('Data/Research Question 2 Final/weekly_data_prepared.csv')
dfWeekly['Date'] = pd.to_datetime(dfWeekly['Date'])
dfWeekly = dfWeekly.set_index(['Region', 'Date'])
dfWeekly['Regioncol']= dfWeekly.index.get_level_values('Region')
dfWeekly = dfWeekly[['mosquito', 'dengue']]

#------------------Set Variables for Estimation----------------------------------
#set mosquito lags
dfProvince['mosquito_lag1'] = dfProvince.groupby('Provincecol')['mosquito'].shift(1)
dfProvince['mosquito_lag2'] = dfProvince.groupby('Provincecol')['mosquito'].shift(2)

#set quantile boundaries
q1 = dfProvince['mosquito'].quantile(.25)
q2 = dfProvince['mosquito'].quantile(.5)
q3 = dfProvince['mosquito'].quantile(.75)

#add quantiles as columns for mosquito and its lags
dfProvince['M_cat1'] = dfProvince['mosquito']< q1
dfProvince['M_cat2'] = (dfProvince['mosquito'] > q1) & (dfProvince['mosquito']< q2) 
dfProvince['M_cat3'] = (dfProvince['mosquito'] > q2) & (dfProvince['mosquito']< q3) 
dfProvince['M_cat4'] = dfProvince['mosquito'] > q3
#lag 1 quantiles
dfProvince['Ml_cat1'] = dfProvince['mosquito_lag1'] < q1
dfProvince['Ml_cat2'] = (dfProvince['mosquito_lag1'] > q1) & (dfProvince['mosquito_lag1']< q2) 
dfProvince['Ml_cat3'] = (dfProvince['mosquito_lag1'] > q2) & (dfProvince['mosquito_lag1']< q3) 
dfProvince['Ml_cat4'] = dfProvince['mosquito_lag1'] > q3
#lag 2 quantiles
dfProvince['Ml2_cat1'] = dfProvince['mosquito_lag2'] < q1
dfProvince['Ml2_cat2'] = (dfProvince['mosquito_lag2'] > q1) & (dfProvince['mosquito_lag1']< q2) 
dfProvince['Ml2_cat3'] = (dfProvince['mosquito_lag2'] > q2) & (dfProvince['mosquito_lag1']< q3) 
dfProvince['Ml2_cat4'] = dfProvince['mosquito_lag2'] > q3

#take log of dengue and add its lag in data frame
dfProvince['log_dengue'] = np.log(dfProvince['Dengue'] + 1)
dfProvince['lag_log_dengue'] = dfProvince['log_dengue'].shift(1)

#-------------------- Model ---------------------------------------------
#first specification
X_spec1 = sm.add_constant(dfProvince.loc[:,['lag_log_dengue','M_cat2','M_cat3', 'M_cat4']])
mod1 = PanelOLS(dfProvince['log_dengue'], X_spec1, entity_effects = True)
res1 = mod1.fit(cov_type = 'clustered')

print(res1)

#second specification with kfold
X_spec2 = sm.add_constant(dfProvince.loc[:,['M_cat2','M_cat3', 'M_cat4','Ml_cat2','Ml_cat3', 'Ml_cat4', 'Ml2_cat2', 'Ml2_cat3','Ml2_cat4']])
mod2 = PanelOLS(dfProvince['log_dengue'], X_spec2, entity_effects = True)
res2 = mod2.fit(cov_type = 'clustered')
print(res2)

#third specification with kfold
X_spec3 = sm.add_constant(dfProvince.loc[:,['lag_log_dengue','M_cat2','M_cat3', 'M_cat4','Ml_cat2','Ml_cat3', 'Ml_cat4', 'Ml2_cat2', 'Ml2_cat3','Ml2_cat4']])
mod3 = PanelOLS(dfProvince['log_dengue'], X_spec3, entity_effects = True)
res3 = mod3.fit(cov_type = 'clustered')
print(res3)

#--------------- k-Fold CV ------------------------------------------------
#kFold can only be performed if no NAs are included, so if lags are added
# kfold is only performed on the observations without NA, constructed as below
dfProvince['nona'] = ~dfProvince['log_dengue'].isna() * ~dfProvince['mosquito'].isna() * ~dfProvince['mosquito_lag1'].isna() * ~dfProvince['mosquito_lag2'].isna()
y = dfProvince['log_dengue'].loc[dfProvince['nona']]
X1 =sm.add_constant(dfProvince.loc[:,['lag_log_dengue','M_cat2','M_cat3','M_cat4']]).loc[dfProvince['nona']]
X2 = sm.add_constant(dfProvince.loc[:,['M_cat2','M_cat3','M_cat4','Ml_cat2','Ml_cat3','Ml_cat4','Ml2_cat2','Ml2_cat3','Ml2_cat4']]).loc[dfProvince['nona']]
X3 = sm.add_constant(dfProvince.loc[:,['lag_log_dengue','M_cat2','M_cat3','M_cat4','Ml_cat2','Ml_cat3','Ml_cat4','Ml2_cat2','Ml2_cat3','Ml2_cat4']]).loc[dfProvince['nona']]

#on these sets the kfold function can be called



