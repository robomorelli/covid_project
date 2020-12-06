import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import leastsq
import datetime
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
import os
import collections

class df_preprocessing():
    def __init__(self, df, datetime_columns = None):
        """ Initialization of the class fit

        :param df: input dataframe or filename
        :param datetime_columns: columns of dates
        :param analysis_columns: columns to analyze
        :type df: pd.DataFrame
        :type columns_datetime: string
        :type start_date: string
        :type end_date: string
        :type prevision: int(string)
        """
        # reading dataframe
        if(isinstance(df,pd.DataFrame)):self.__df = df 
        if(isinstance(datetime_columns,str)):self.__df.set_index('DATA', inplace = True)
        # initializing inherited classes
#         super(plotter, self).__init__() #inheredite from other classes and use delf.plotte.method
        
    def df_show(self):
        return self.__df.head()
    
    
class fitter():
    def __init__(self, df, datetime_columns = None, columns_analysis= None, start_date= None
                 , end_date= None, prevision= None):
        """ Initialization of the class fit

        :param df: input dataframe or filename
        :param datetime_columns: columns of dates
        :param analysis_columns: columns to analyze
        :type df: pd.DataFrame
        :type columns_datetime: string
        :type start_date: string
        :type end_date: string
        :type prevision: int(string)
        """
        super(df_preprocessing, self).__init__() #inheredite from other classes and use delf.plotte.method
        # reading dataframe
        if(isinstance(df,pd.DataFrame)):self.__df = df 
        if(isinstance(datetime_columns,str)):self.__df.set_index('DATA', inplace = True)
        # initializing inherited classes

        
#     def df_show(self):
#         return self.__df.head()
    
#     def par_n_fit(model_name, df, columns, end_fit, n_days, p0=None, moving_average = False,
#                      plot = False, weight = False):
        
#     def par_n_fit(model_name, df, columns, end_fit, n_days, p0=None, moving_average = False,
#                      plot = False, weight = False):

#         end_fit = datetime.datetime.strptime(end_fit, "%Y-%m-%d")
#         start_fit = end_fit - datetime.timedelta(n_days-1)
#         print(start_fit.date(),end_fit.date())

#         if moving_average:
#             label_average = moving_average[0]
#             period = moving_average[1]
#             ma_summary = [label_average,period]

#             if label_average == 'uniform':
#                 df_averaged = df.loc[:end_fit][columns]\
#                               .rolling(window=period).mean()
#                 series = df_averaged.loc[start_fit:end_fit]
#                 series = np.round(series, decimals=0) 

#             elif label_average == 'ewm':
#                 df_averaged = df.loc[:end_fit][columns]\
#                               .ewm(span=period).mean()
#                 series = df_averaged.loc[start_fit:end_fit]
#                 series = np.round(series, decimals=0) 

#             elif label_average == 'wma':
#                 weights = np.arange(1,period+1)
#                 df_averaged = df.loc[:end_fit][columns].rolling(period)\
#                           .apply(lambda counts: np.dot(counts, weights)/weights.sum(), 
#                                  raw=True)
#                 series = df_averaged.loc[start_fit:end_fit]
#                 series = np.round(series, decimals=0)      
#         else:
#             label_average = False
#             series = df.loc[start_fit:end_fit][columns] 
#             ma_summary = label_average

#         dates_fit = series.index.date

#         x_range = np.array([x for x in range(len(series))])
#         y_values = np.array(series)

#         func_fit = map_func[model_name]['fitter']
#         func_plot = map_func[model_name]['plotter']

#         if p0!= None:
#             if weight:
#                 sigma = np.array([np.sqrt(n) for n in list(y_values)])
#                 par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, p0, sigma=sigma,absolute_sigma=True
#                                            ,maxfev=1000000)
#             else:
#                 par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, p0
#                                               ,maxfev=1000000)
#         else:   
#             if weight:
#                 sigma = np.array([np.sqrt(n) for n in list(y_values)])
#                 par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, sigma=sigma,absolute_sigma=True
#                                            ,maxfev=1000000)
#             else:
#                 par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values
#                                               ,maxfev=1000000)

#         ddof = len(par[0])

#         if model_name == 'exponential':
#             dT = np.log(2)/par[0][1]
#         else:
#             dT = None

#         pars = par[0]
#         sigma_pars = np.sqrt(np.diag(par[1]))
#         expected_values = func_plot(x_range, pars)
#         chi, p = chisquare(y_values, expected_values,ddof)
#         print(chi, p)

#         if plot:

#             plt.figure(figsize=(6, 4))
#             plt.scatter(x_range, y_values, label='Data')
#             plt.plot(x_range, func_plot(x_range, pars),
#                      label='Fitted function')
#             plt.legend(loc='best')
#             plt.xlabel('day')
#             plt.xticks(x_range, dates_fit,rotation='vertical')

#             plt.show()

#         dic = {}
#         dic['parameters_fit'] = pars
#         dic['sigma_pars'] = sigma_pars
#         dic['dT'] = dT
#         dic['fitted_day'] = n_days
#         dic['end_fit'] = end_fit.strftime("%Y-%m-%d")
#         dic['fitted_dates'] = dates_fit
#         dic['fitted_series'] = series
#         dic['moving_average'] = ma_summary
#         dic['chi_2'] = [chi,p]
#         dic['model'] = model_name

#         return dic, x_range, dates_fit, y_values
    
    

    
########################################

def par_n_fit(model_name, df, columns, end_fit, n_days, p0=None, moving_average = False,
                 plot = False, weight = False):
    
    end_fit = datetime.datetime.strptime(end_fit, "%Y-%m-%d")
    start_fit = end_fit - datetime.timedelta(n_days-1)
    print(start_fit.date(),end_fit.date())

    if moving_average:
        label_average = moving_average[0]
        period = moving_average[1]
        ma_summary = [label_average,period]
        
        if label_average == 'uniform':
            df_averaged = df.loc[:end_fit][columns]\
                          .rolling(window=period).mean()
            series = df_averaged.loc[start_fit:end_fit]
            series = np.round(series, decimals=0) 
            
        elif label_average == 'ewm':
            df_averaged = df.loc[:end_fit][columns]\
                          .ewm(span=period).mean()
            series = df_averaged.loc[start_fit:end_fit]
            series = np.round(series, decimals=0) 
                     
        elif label_average == 'wma':
            weights = np.arange(1,period+1)
            df_averaged = df.loc[:end_fit][columns].rolling(period)\
                      .apply(lambda counts: np.dot(counts, weights)/weights.sum(), 
                             raw=True)
            series = df_averaged.loc[start_fit:end_fit]
            series = np.round(series, decimals=0)      
    else:
        label_average = False
        series = df.loc[start_fit:end_fit][columns] 
        ma_summary = label_average
        
    dates_fit = series.index.date

    x_range = np.array([x for x in range(len(series))])
    y_values = np.array(series)
    
    func_fit = map_func[model_name]['fitter']
    func_plot = map_func[model_name]['plotter']
    
    if p0!= None:
        if weight:
            sigma = np.array([np.sqrt(n) for n in list(y_values)])
            par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, p0, sigma=sigma,absolute_sigma=True
                                       ,maxfev=1000000)
        else:
            par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, p0
                                          ,maxfev=1000000)
    else:   
        if weight:
            sigma = np.array([np.sqrt(n) for n in list(y_values)])
            par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values, sigma=sigma,absolute_sigma=True
                                       ,maxfev=1000000)
        else:
            par = scipy.optimize.curve_fit(func_fit,  x_range,  y_values
                                          ,maxfev=1000000)

    ddof = len(par[0])
    
    if model_name == 'exponential':
        dT = np.log(2)/par[0][1]
    else:
        dT = None
    
    pars = par[0]
    sigma_pars = np.sqrt(np.diag(par[1]))
    expected_values = func_plot(x_range, pars)
    chi, p = chisquare(y_values, expected_values,ddof)
    print(chi, p)
    
    if plot:
    
        plt.figure(figsize=(6, 4))
        plt.scatter(x_range, y_values, label='Data')
        plt.plot(x_range, func_plot(x_range, pars),
                 label='Fitted function')
        plt.legend(loc='best')
        plt.xlabel('day')
        plt.xticks(x_range, dates_fit,rotation='vertical')

        plt.show()
        
    dic = {}
    dic['parameters_fit'] = pars
    dic['sigma_pars'] = sigma_pars
    dic['dT'] = dT
    dic['fitted_day'] = n_days
    dic['end_fit'] = end_fit.strftime("%Y-%m-%d")
    dic['fitted_dates'] = dates_fit
    dic['fitted_series'] = series
    dic['moving_average'] = ma_summary
    dic['chi_2'] = [chi,p]
    dic['model'] = model_name
    
    return dic, x_range, dates_fit, y_values