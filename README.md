# The Importance of Spatio-Temporal Granularity for Modelling Philippine Dengue Incidence, using Mosquito Density and Weather Data
### 510-Group2, Seminar Case Studies Erasmus University Rotterdam

## Abstract
This study evaluates the forecasting performance of mosquito density for dengue incidence in the Philippines. Methodological as well as data-related research questions are considered. The importance of granularity in dengue incidence data is evaluated. Dengue forecasting performance is found to be much better for data sets that are spatially more granular, or temporally more granular, compared to a month-region specific dengue incidence data set. A fixed effects model is compared to a Seasonal Autoregressive Integrated Moving Average model on different data sets. Only limited evidence is found to support the extensive autoregressive components of the seasonal autoregressive model, with no significant evidence of seasonality in dengue incidence. On the provincial level in 2015 mosquito density has a significant effect on dengue incidences. Mosquito density in the highest of four quantiles is associated with a 46.2\% higher incidence of dengue relative to the lowest quantile. As reliable data on mosquito density is hardly available, the predicting performance of mosquito density is compared to that of meteorological data. Multiple weather variables are shown to have a significant effect on dengue incidence and forecasting performance is found to be equivalent to slightly improved.

## Data
For this study, we have data about the number of dengue incidences, the mosquito density and some weather variables. The weather data is collected by the Google Earth Engine (https://developers.google.com/earth-engine/datasets/catalog/). Due to privacy concerns, not all data can be shared in this GitHub. Dengue incidences on monthly level can be found here: https://www.doh.gov.ph/statistics. Mosquito density can be found here: http://dengue.pchrd.dost.gov.ph/. 

## Software Explanation
The code needed for the analysis in this paper is given in four files, for each of the three research questions and the datapreparation. Packages that are needed for the analysis, are given in the requirements.txt file.

### Data preparation
In this file, the code is provided for the data preparation. The mosquito data here is prepared to a full data set on month region level.
Three different data sets, region-month, weekly and province, are imputed all in the same way. All data sets are first loaded with corresponding geographical data.

First weighted kNN is performed: for each observation, missing or not, its k (georgraphically) nearest neighbours are found and a count-weighted average of these neighbours is calculated. This gives a full data frame of imputed values (also for the non-missing). This is done for all three data sets.

Then the missing values in the original dataframe are replaced by the imputed values such that a full dataframe is created. 


### Research Question 1
In this file, the code is provided for the first research question: "Does the SARIMA model achieve superior performance to a fixed effects model in modelling dengue incidence from mosquito density?". 
This analysis is done on month-region level. As this question is answered with quantiles for mosquito density, these are first made in the code. 

Then, the results for the SARIMA model are given from Table 3 in the paper. To get the optimal values we try different hyperparameter settings, implement the SARIMA model per region, and summarize the AIC and BIC over each region. Then, using these hyperparameters, the SARIMA model is implemented aggregated over regions. 

Last, the code for the results for the fixed effects, as stated in Table 4 in the paper, is given. The fixed effects model is done for three specification, as is stated in the paper. These three specifications can be compared using R^2, AIC, BIC, and OOS R^2 and OOS MAE using k-fold Cross Validation.  


### Research Question 2
In this file, the code is provided for the second research question: "What is the importance of spatial- and temporal granularity in accurately modelling dengue incidence using mosquito density?". This analysis is done on weekly-region level and month-province level. First, the weekly and province level dataset are loaded. The code works out the province-level model but can also be used for the weekly-level model. 

Then, variables are added to the dataframe; quantiles of mosquito density, its first and second lag, and the lag of log of dengue. 

Last, the code for the three different specifications of the fixed effects is given resulting in Table 5 of the paper. If the weekly-level dataset is selected in the beginning, this will result in the output of Table 6.

### Research Question 3
In this file, the code is provided for the third research question: "Does a model explaining dengue incidence based on weather data outperform a model based on mosquito density?". This analysis is done on month-province level.

First, the prepared province level weather dataset is loaded. Preparations that are made can be found in the Section 4.5 and 5.2.4. 

Then, the code for the three different specifications of the fixed effects is given. This results in the output of Table 7. Last, AIC is calculated from the output of the different model specifications.  

### k-fold Cross-Validation 
In this file, the code for the k-fold Cross-Calidation is shown. For the results that are stated in the paper, k is chosen as 10. The function ouotputs the Out-Of-Sample (OOS) R^2 and Mean-Absolute-Error (MAE). 
