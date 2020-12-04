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
import sys

sys.path.append('../..')
from fit_utilities import *


    
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

def plotter(x0,y0, x_val, y_val, x_fit_val = None, y_fit_val = None, show = True):
    
    if isinstance(x_val, list):
        plt.figure(figsize=(12, 8))
        if show:
            series = zip(x0, y0, x_val, y_val, x_fit_val, y_fit_val)
            for x_0, y_0, x, y, x_fit, y_fit in series:
                plt.scatter(x_0, y_0, label='Data with test')
#                 plt.scatter(x, y, label='Data')
                plt.plot(x_fit, y_fit,label='Fitted function')
                plt.legend(loc='best')
                plt.xlabel('range')
            plt.show()
        else:
            series = zip(x_val, y_val, x_fit_val, y_fit_val)
            for x, y, x_fit, y_fit in series:
                plt.scatter(x, y, label='Data')
                plt.plot(x_fit, y_fit,
                         label='Fitted function')
                plt.legend(loc='best')
                plt.xlabel('range')
            plt.show()

    else:
        plt.figure(figsize=(12, 8))
        if show:
            plt.scatter(x0, y0, label='Data')
        plt.scatter(x_val, y_val, label='Data')
        plt.plot(x_fit_val, y_fit_val,
                 label='Fitted function')
        plt.legend(loc='best')
        plt.xlabel('range')
    #     plt.xticks(x, dates_fit, rotation='vertical')
        plt.show()


class TableFiltering():
    
    def __init__(self):
        """ Initialization of the class table filtering: fimple filtering process
        """
#         if(isinstance(df,pd.DataFrame)):self.df = df.copy()
    
    def preprocessing(self, df, select = None, cuts = None, return_original=False):
        '''apply filters'''
        if select is not None:
            for k,v in select.items():
                if isinstance(v, list):
                    df = df[df[k].isin(v)]
                else:
                    df = df[df[k].isin([v])]

        if cuts is not None:
            for k,v in cuts.items():
                if 'and' in v:
                    condition = v.replace(' ','')
                    condition = condition.split('and')
                    for con in condition:
                        print('{} {} {}'.format(k,con[0],con[1:]))
                        df = df.query('{} {} {}'.format(k,con[0],con[1:]))
                elif 'or' in v:
                    condition = v.replace(' ','')
                    condition = condition.split('or')
                    df_filtered = pd.DataFrame(columns = df.columns)
                    for con in condition:
                        df_filtered = df_filtered.append(df.query('{} {} {}'.format(k,con[0],con[1:])))
                    df = df_filtered.drop_duplicates(df_filtered.columns)
                else:
                    con = v.replace(' ','')
                    print(con)
                    if con[0] == '=':
                        comp = '=='
                        cond = con[1:]
                        df = df.query('{} {} {}'.format(k,comp,cond))
                    elif con[0:2] in ['==','eq']:
                        comp = '=='
                        cond = con[2:]

                        df = df.query('{} {} {}'.format(k,comp,cond))
                        print('{} {} {}'.format(k,comp,cond))
                    else:
                        df = df.query('{} {} {}'.format(k,con[0],con[1:]))
#             else:
#                 TypeError('cuts is not a dictionary type')
        if return_original:
            return df, df
        else:        
            return df