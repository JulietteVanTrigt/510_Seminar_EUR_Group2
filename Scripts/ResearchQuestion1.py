"""
Code for the analysis and results regarding Research Question 1
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from itertools import product
from warnings import filterwarnings
from linearmodels import PanelOLS
from kfoldCV import kfoldfun

path = ".../Dengue_modelling/"
df_long = pd.read_csv(path + "Data/Research Question 1 Final/Long_KNN_imputed_weighted_outliers_removed.csv")
df = df_long[['Region','Date','count','dengue','mosquito']] 
regions = df.Region.unique().tolist()

# =============================================================================
# Table 3: SARIMA(2,1,1)(1,0,0)12 for 2013-2016, aggregated over region
# =============================================================================
agg_df = df_long[['Region','Date','count','dengue','mosquito']].groupby('Date').mean()
agg_df['log_dengue'] = np.log(agg_df['dengue']+1)
MD_25 = agg_df['mosquito'].describe()[4]
MD_50 = agg_df['mosquito'].describe()[5]
MD_75 = agg_df['mosquito'].describe()[6]
agg_df['M_cat1'] = agg_df['mosquito'] <= MD_25
agg_df['M_cat2'] = (agg_df['mosquito'] <= MD_50) & (agg_df['mosquito'] > MD_25)
agg_df['M_cat3'] = (agg_df['mosquito'] <= MD_75) & (agg_df['mosquito'] > MD_50)
agg_df['M_cat4'] = (agg_df['mosquito'] > MD_75)

parameters = []
p = q = d = range(0, 3)
s = range(0, 2)
pdq = list(product(p, d, q))
m = (3, 6, 12)
D = range(0, 1)
seasonal_pdq= [(x[0], x[1], x[2], x[3]) for x in list(product(s, D, s, m))] # Gives 324 iterations
filterwarnings("ignore")
i = 0
for it_order in pdq:
    for it_seasonal_order in seasonal_pdq:
        print(it_order,it_seasonal_order)
        print(str(i) + ' out of ' + str(len(pdq)*len(seasonal_pdq)))
        i += 1
        reg_endog = agg_df['dengue']
        reg_exog  = agg_df[['M_cat2','M_cat3','M_cat4']].astype(int)
        model = sm.tsa.statespace.SARIMAX(endog = reg_endog,
                              exog = reg_exog, 
                              order = it_order, 
                              seasonal_order= it_seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        results = model.fit()
        aic = results.aic
        bic = results.bic
        parameters.append([it_order,it_seasonal_order,aic,bic])

result_table = pd.DataFrame(parameters)
result_table.columns = ['parameters','parameters_seasonal','aic','bic']
result_table_aic = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
result_table_bic = result_table.sort_values(by='bic', ascending=True).reset_index(drop=True)        
result_table_aic.head()
result_table_bic.head()

order_final = list(result_table_bic['parameters'][0])
seasonal_order_final = list(result_table_bic['parameters_seasonal'][0])    
#order_final = (2,1,1)
#seasonal_order_final = (1,0,0,12)

reg_endog = agg_df['log_dengue']
reg_exog  = agg_df[['M_cat2','M_cat3','M_cat4']].astype(int)
model = sm.tsa.statespace.SARIMAX(endog = reg_endog,
                                  exog = reg_exog, 
                                  order = order_final, 
                                  seasonal_order= seasonal_order_final,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
res = model.fit()
print(res.summary())


# =============================================================================
# Table 4: Results fixed effects monthly from 2013-2016, aggregated over regions
# =============================================================================
df = df_long[['Region','Date','count','dengue','mosquito']] 
MD_25 = df['mosquito'].describe()[4]
MD_50 = df['mosquito'].describe()[5]
MD_75 = df['mosquito'].describe()[6]
df['M_cat1'] = df['mosquito'] <= MD_25
df['M_cat2'] = (df['mosquito'] <= MD_50) & (df['mosquito'] > MD_25)
df['M_cat3'] = (df['mosquito'] <= MD_75) & (df['mosquito'] > MD_50)
df['M_cat4'] = (df['mosquito'] > MD_75)
df['Ml_cat1'] = df.groupby('Region')['M_cat1'].shift(1)
df['Ml_cat2'] = df.groupby('Region')['M_cat2'].shift(1)
df['Ml_cat3'] = df.groupby('Region')['M_cat3'].shift(1)
df['Ml_cat4'] = df.groupby('Region')['M_cat4'].shift(1)
df['Ml2_cat1'] = df.groupby('Region')['M_cat1'].shift(2)
df['Ml2_cat2'] = df.groupby('Region')['M_cat2'].shift(2)
df['Ml2_cat3'] = df.groupby('Region')['M_cat3'].shift(2)
df['Ml2_cat4'] = df.groupby('Region')['M_cat4'].shift(2)
df['log_dengue'] = np.log(df['dengue']+1)
df['lag_dengue'] = df.groupby('Region')['dengue'].shift(1)
df['log_lag_dengue'] = np.log(df['lag_dengue']+1)

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Region','Date'])

def Panel_output(endo,exog):
    X = sm.add_constant(df.loc[:,exog])
    mod = PanelOLS(endo, X, entity_effects = True)
    res = mod.fit(cov_type='clustered')
    print(res)
    return(res.loglik,exog,res)

S1 = Panel_output(df['log_dengue'],['log_lag_dengue',
                                    'M_cat2','M_cat3','M_cat4'])
S2 = Panel_output(df['log_dengue'],['M_cat2','M_cat3','M_cat4',
                                    'Ml_cat2','Ml_cat3','Ml_cat4',
                                    'Ml2_cat2','Ml2_cat3','Ml2_cat4'])
S3 = Panel_output(df['log_dengue'],['log_lag_dengue',
                                    'M_cat2','M_cat3','M_cat4',
                                    'Ml_cat2','Ml_cat3','Ml_cat4',
                                    'Ml2_cat2','Ml2_cat3','Ml2_cat4'])
# AIC and BIC
AIC_1 = 2 * (len(S1[1]) - S1[0])
AIC_2 = 2 * (len(S2[1]) - S2[0])
AIC_3 = 2 * (len(S3[1]) - S3[0])

# Specification 1
print("Output Specification 1:")
print(S1[2])
print(AIC_1)
df1 = df[pd.notnull(df['lag_dengue'])]
exog = ['log_lag_dengue',
        'M_cat2','M_cat3','M_cat4']
X = sm.add_constant(df1.loc[:,exog])    
y = df1['log_dengue']
kfoldfun(y, X, 10) 

# Specification 2
print("Output Specification 2:")
print(S2[2])
print(AIC_2)
df2 = df[pd.notnull(df['Ml2_cat2'])]
exog = ['log_lag_dengue',
        'M_cat2','M_cat3','M_cat4']
X = sm.add_constant(df2.loc[:,exog])    
y = df2['log_dengue']
kfoldfun(y, X, 10) 

# Specification 3
print("Output Specification 3:")
print(S3[2])
print(AIC_3)
df3 = df[pd.notnull(df['Ml2_cat2'])]
exog = ['log_lag_dengue',
        'M_cat2','M_cat3','M_cat4']
X = sm.add_constant(df3.loc[:,exog])    
y = df3['log_dengue']
kfoldfun(y, X, 10) 
