#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# RESEARCH QUESTION 3
# This code produces the output of the third research question:
# "Does a model predicting dengue incidence based on weather data outperform 
#  a model based on mosquito density?"
# =============================================================================

#Load packages
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS
from kfoldCV import kfoldfun

#Load the prepared weather data 
#Preparation is described in the papers
path = "..." #define path
df = pd.read_csv(path + "/Data/Prov2015_weatherNew_norm_logdengue2.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Province', 'Date'])

# =============================================================================
# Variables in DataFrame
# =============================================================================
#W1: hourly precipitation rate adjested to Rain Gaug    (mm/hr)
#W2: land surface temperature day, scale = 0.02         (K)
#W3: land surface temperature night, scale = 0.02       (K)
#W4: enhanced vegatation index 
#W5: humidity                                           (kg/kg)
#W6: Total Percipitation rate                           (kg/(m2*s))
#W7: Soil Moisture 0-10cm underground                   (m3/m3)
#W8: Soil Temperature 0-10cm underground                (K)
#W9: air temperature                                    (K)
#W10: wind speed                                        (m/s)
#M_cat   : Mosquito category
#Ml_cat  : Mosquito category lagged once 
#Ml2_cat : Mosquito category lagged twice 
#W.._lag : W.. lagged
#W.._cat : Categories for W..
#MD_ord  : Ordinal Mosquito density for MNL
# =============================================================================

def Panel_output(endo,exog):
    X = sm.add_constant(df.loc[:,exog])
    mod = PanelOLS(df['log_dengue'], X, entity_effects = True)
    res = mod.fit(cov_type='clustered')
    print(res)
    return(res.loglik,exog,res)

# =============================================================================
# Three model specifications (restults from Section 6.3)
# =============================================================================
#Specification 1
exog1 = ['log_lag_dengue',
        'W1l_cat2','W1l_cat3','W1l_cat4',
        'W2l_cat2','W2l_cat3','W2l_cat4',
        'W10l_cat2','W10l_cat3','W10l_cat4']
S1 = Panel_output(df['log_dengue'],exog1)

#Sepcification 2
exog2 = ['W1l_cat2','W1l_cat3','W1l_cat4',
        'W2l_cat2','W2l_cat3','W2l_cat4',
        'W10l_cat2','W10l_cat3','W10l_cat4',
        'W1ll_cat2','W1ll_cat3','W1ll_cat4',
        'W2ll_cat2','W2ll_cat3','W2ll_cat4',
        'W10ll_cat2','W10ll_cat3','W10ll_cat4']
S2 = Panel_output(df['log_dengue'],exog2)

#Specification 3
exog3 = ['log_lag_dengue',
        'W1l_cat2','W1l_cat3','W1l_cat4',
        'W2l_cat2','W2l_cat3','W2l_cat4',
        'W10l_cat2','W10l_cat3','W10l_cat4',
        'W1ll_cat2','W1ll_cat3','W1ll_cat4',
        'W2ll_cat2','W2ll_cat3','W2ll_cat4',
        'W10ll_cat2','W10ll_cat3','W10ll_cat4']
S3 = Panel_output(df['log_dengue'],exog3)

# AIC and BIC
AIC_1 = 2 * (len(S1[1]) - S1[0])
AIC_2 = 2 * (len(S2[1]) - S2[0])
AIC_3 = 2 * (len(S3[1]) - S3[0])

print("Output Specification 1:")
print(S1[2])
print(AIC_1)
print("Output Specification 2:")
print(S2[2])
print(AIC_2)
print("Output Specification 3:")
print(S3[2])
print(AIC_3)

# =============================================================================
# k-fold Cross-Validation, remove NA in the lags for specification 1&3
# =============================================================================
df = pd.read_csv(path + '/Data/Prov2015_weatherNew_norm_logdengue2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Province', 'Date'])

y2 = df['log_dengue']
X2 = sm.add_constant(df.loc[:,exog2])

df = df[pd.notnull(df['log_lag_dengue'])]
y = df['log_dengue']
X1 = sm.add_constant(df.loc[:,exog1])
X3 = sm.add_constant(df.loc[:,exog3]) 

kfoldfun(y, X1)
kfoldfun(y2, X2)
kfoldfun(y, X3)
