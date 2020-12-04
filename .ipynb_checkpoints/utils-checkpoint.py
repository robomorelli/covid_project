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

dic = {'linear':{'func_fit':lin_func_fit}}

# def exp_func(t, a, b):
#     return a*np.exp(b*t)

# def exp1_func(t, b):
#     return 1-np.exp(b*t)

# def lin_func(t, intercept, coeff):
#     return intercept + coeff * t

# def gompertz_func_fit_old(x,a,b,c):
#     return c*np.exp(-b*np.exp(-x/a)) 

# def gompertz_func_old(x, pars):
#     a = pars[0]
#     b = pars[1]
#     c = pars[2]
#     return c*np.exp(-b*np.exp(-x/a)) 

def gompertz_func_fit_alfa(x,a,b,c):
    return c*np.exp(-a*np.exp(-b*x)) 

def gompertz_func_alfa(x, pars):
    a = pars[0]
    b = pars[1]
    c = pars[2]
    return c*np.exp(-a*np.exp(-b*x)) 

def gompertz_func_fit(x,k,N0,r):
    return k*np.exp(np.log(N0/k)*np.exp(-r*x)) 

def gompertz_func(x, pars):
    k = pars[0]
    N0 = pars[1]
    r = pars[2]
    return k*np.exp(np.log(N0/k)*np.exp(-r*x)) 

################################################
###############################################
##############################################

def gompertz_der_func_fit_alfa(x,a,b,c):
    return a*b*c*np.exp(-a*np.exp(-b*x) - b*x) 

def gompertz_der_func_alfa(x, pars):
    a = pars[0]
    b = pars[1]
    c = pars[2]
    return a*b*c*np.exp(-a*np.exp(-b*x) - b*x) 

def gompertz_der_func_fit(x,k,N0,r):
    return -r*np.log(N0/k)*k*np.exp(np.log(N0/k)\
                                    *np.exp(-r*x))*(np.exp(-r*x))

def gompertz_der_func(x, pars):
    k = pars[0]
    N0 = pars[1]
    r = pars[2]
    return -r*np.log(N0/k)*k*np.exp(np.log(N0/k)\
                                    *np.exp(-r*x))*(np.exp(-r*x))

################################################
###############################################
##############################################
def logistic_func_fit_alfa(x,N,xmax,b):
    return N/(1+np.exp(-(x-xmax)/b))

def logistic_func_alfa(x, pars):
    N = pars[0]
    xmax = pars[1]
    b = pars[2]
    return N/(1+np.exp(-(x-xmax)/b))


def logistic_func_fit(x,k,N0,r):
    return k/(1+((k-N0)/N0)*np.exp(-r*x))

def logistic_func(x, pars):
    k = pars[0]
    N0 = pars[1]
    r = pars[2]
    return k/(1+((k-N0)/N0)*np.exp(-r*x))

################################################
###############################################
##############################################

def logistic_der_func_fit_alfa(x,N,xmax,b):
    return ((N/b)*np.exp(-(x-xmax)/b))/((1+np.exp(-(x-xmax)/b))**2)

def logistic_der_func_alfa(x, pars):
    N = pars[0]
    xmax = pars[1]
    b = pars[2]
    return ((N/b)*np.exp(-(x-xmax)/b))/((1+np.exp(-(x-xmax)/b))**2)


def logistic_der_func_fit(x,k,N0,r):
    return (k*r*((k-N0)/N0)*np.exp(-r*x))/((1+((k-N0)/N0)*np.exp(-r*x))**2)

def logistic_der_func(x, pars):
    k = pars[0]
    N0 = pars[1]
    r = pars[2]
    return (k*r*((k-N0)/N0)*np.exp(-r*x))/((1+((k-N0)/N0)*np.exp(-r*x))**2)

################################################
###############################################
##############################################

def logistic_gen_func_fit(x,a,m,n,tau):
    return a*( (1+m*np.exp(-x/tau))/ (1+n*np.exp(-x/tau)) )

def logistic_gen_func(x, pars):
    a = pars[0]
    m = pars[1]
    n = pars[2]
    tau = pars[3]
    return a*( (1+m*np.exp(-x/tau))/ (1+n*np.exp(-x/tau)) )

def logistic_gen_der_func_fit(x,a,m,n,tau):
    return (a/(1+n*np.exp(-x/tau))**2) * \
                    ( (n/tau)*np.exp(-x/tau) - (m/tau)*np.exp(-x/tau) )

def logistic_gen_der_func(x, pars):
    a = pars[0]
    m = pars[1]
    n = pars[2]
    tau = pars[3]
    return (a/(1+n*np.exp(-x/tau))**2) * \
                    ( (n/tau)*np.exp(-x/tau) - (m/tau)*np.exp(-x/tau) )

def lin_func_fit(x, a,b):
    return a + b * x

def lin_func(x, pars):
    intercept = pars[0]
    coeff = pars[1]
    return intercept + coeff * x

def exp_func_fit(x, a,b):
    return a*np.exp(b*x)

def exp_func(x, pars):
    a = pars[0]
    b = pars[1]
    return a*np.exp(b*x)



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

#     a,b,c = par[0]
#     sa,sb,sc = np.sqrt(np.diag(par[1]))

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





map_func = {'linear':{'fitter':lin_func_fit,
                     'plotter':lin_func,
                     'category':par_n_fit,
                     'p0':None},
            'exponential':{'fitter':exp_func_fit,
                     'plotter':exp_func,
                    'category':par_n_fit,
                    'p0':None},
            
            'logistic_alfa':{'fitter':logistic_func_fit_alfa,
                     'plotter':logistic_func_alfa,
                      'category':par_n_fit,
                       'p0':None}, #[3,50,5000]
            
            'logistic':{'fitter':logistic_func_fit,
                     'plotter':logistic_func,
                      'category':par_n_fit,
                       'p0':None}, #[3,50,5000]
            
            'logistic_der_alfa':{'fitter':logistic_der_func_fit_alfa,
                     'plotter':logistic_der_func_alfa,
                      'category':par_n_fit,
                       'p0':None}, #[3,50,5000]
            
            'logistic_der':{'fitter':logistic_der_func_fit,
                     'plotter':logistic_der_func,
                      'category':par_n_fit,
                       'p0':None},
            
           'logistic_gen':{'fitter':logistic_gen_func_fit,
                     'plotter':logistic_gen_func,
                      'category':par_n_fit,
                       'p0':None},
           'logistic_gen_der':{'fitter':logistic_gen_der_func_fit,
                     'plotter':logistic_gen_der_func,
                      'category':par_n_fit,
                       'p0':None},
            'gompertz':{'fitter':gompertz_func_fit,
                     'plotter':gompertz_func,
                      'category':par_n_fit,
                       'p0':None},
            'gompertz_der':{'fitter':gompertz_der_func_fit,
                     'plotter':gompertz_der_func,
                      'category':par_n_fit,
                       'p0':None},
            'gompertz_alfa':{'fitter':gompertz_func_fit_alfa,
                     'plotter':gompertz_func_alfa,
                      'category':par_n_fit,
                       'p0':[10,1,100]},
            'gompertz_der_alfa':{'fitter':gompertz_der_func_fit_alfa,
                     'plotter':gompertz_der_func_alfa,
                      'category':par_n_fit,
                       'p0':[10,1,100]},
                 }


def latex_report(df, path, name, prevision, dic={}):
    report = {}
    report['DATA'] = df.index.date
    if ('mean' in df.columns) & ('error' in df.columns): 
        
        for col in df.columns:
            col_id = col.split('_')[0].split('_')[0]
            
            if col_id in dic.keys():
                report[dic[col_id]] = ['{:6.0f}'.format(b) for b in df[col].values]
            else:  
                if (col != 'mean') & (col != 'error') & (col != 'DATA'):
                    report[col] = ['{:6.0f}'.format(b) for b in df[col].values]
                
        report['mean  (\u00B1 error)'] = ['{:6.0f} (\u00B1 {:6.0f})'.format(m, e) 
                                    for m,e in zip(df['mean'].values,df['error'].values)]
    
    else:
        for col in df.columns:
            col_id = col.split('_')[0].split('_')[0]
            
            if col_id in dic.keys():
                report[dic[col_id]] = ['{:6.0f}'.format(b) for b in df[col].values]
            else:  
                if col != 'DATA':
                    report[col] = ['{:6.0f}'.format(b) for b in df[col].values]

    col_format = ''    
    col_format = 'l' + 'c'*(len(report.keys())-1)
    df_latex = pd.DataFrame(report)

    rep_lat = df_latex.to_latex(index=False,column_format=col_format) 

    file = open(path + "{}_on_{}_days.txt".format(name,prevision),"w") 
    file.writelines(rep_lat) 
    file.close() #to change file access modes 
    
    
def min_max_date(min_origin, max_origin, date_list):
    
    min_date = datetime.datetime.strptime(min_origin, "%Y-%m-%d")
    for date in date_list:
        temp = datetime.datetime.strptime(date, "%Y-%m-%d")
        if temp <= min_date:
            min_date = temp
            
    max_date = datetime.datetime.strptime(max_origin, "%Y-%m-%d")
    for date in date_list:
        temp = datetime.datetime.strptime(date, "%Y-%m-%d")
        if temp >= max_date:
            max_date = temp
            
    return min_date, max_date

def fit_range(end_fit_list, n_fitted_days, actual_previsions):
    
    min_date_list = []
    for date, nfd in zip(end_fit_list, n_fitted_days):
        for d in nfd:
            min_date_list.append(datetime.datetime.strptime(date, "%Y-%m-%d")- datetime.timedelta(d - 1))
            
    max_date_list = []
    for date, nfd in zip(end_fit_list, actual_previsions):
        for d in nfd:
            max_date_list.append(datetime.datetime.strptime(date, "%Y-%m-%d")\
                                     + datetime.timedelta(int(d)))
            
    return min(min_date_list), max(max_date_list)

def data_extraction(df, analysis, min_start_plot, max_end_fit,upper_limit, exclude_test):
        
    if (upper_limit!=None) & (not exclude_test): #enter the upper limit over the exclude unobs
        print('upper_limit')       
        all_data = df.loc[min_start_plot:upper_limit,[analysis]].values #all data that could be used
        all_data_dates = df.loc[min_start_plot:upper_limit,[analysis]].index.date
        all_data_range = [i for i in range(len(all_data))]
        all_data_df = df.loc[min_start_plot:upper_limit,[analysis]]

    elif exclude_test:
        all_data = df.loc[min_start_plot:max_end_fit,[analysis]].values #all data that could be used
        all_data_dates = df.loc[min_start_plot:max_end_fit,[analysis]].index.date
        all_data_range = [i for i in range(len(all_data))]
        all_data_df = df.loc[min_start_plot:max_end_fit,[analysis]]

    else:  
        all_data = df.loc[min_start_plot:,[analysis]].values #all data that could be used
        all_data_dates = df.loc[min_start_plot:,[analysis]].index.date
        all_data_range = [i for i in range(len(all_data))]
        all_data_df = df.loc[min_start_plot:,[analysis]]
        
    return all_data, all_data_dates, all_data_range, all_data_df

def range_comparison(max_range_total, all_data_range):
    
    if max(max_range_total) >= max(all_data_range):
        max_range_total = max_range_total
    else:
        max_range_total = all_data_range
        
    return max_range_total

def disambiguation_date(end_fit_list):
    flag = False
    end_fit_list_dis = end_fit_list.copy()

    while any(np.array(list(collections.Counter(end_fit_list_dis).values()))>1):
        multiple_date = []
        for d, count in collections.Counter(end_fit_list_dis).items():
            if count > 1:
                multiple_date.append({'date':d,
                                     'times':count,
                                     'start_in_list':end_fit_list_dis.index(d)})
                
        flag = True

        for md in multiple_date[0:1]:
            d = md['date']
            count = md['times']
            sl = md['start_in_list']
            new_dates = [d + '_{}'.format(i) for i in range(1, count+1)]
            [end_fit_list_dis.remove(d) for i in range(count)]

            for i in range(count):
                end_fit_list_dis.insert(sl+i, new_dates[i])
                
    return end_fit_list_dis, flag 

# def fill_from_date(summary, fill_from_this_date, actual_max_range):
            
#     fill_dates = fill_from_this_date.copy()

#     for fill_d in fill_dates:
#         series_fill = {}
#         for ix, k in enumerate(summary.keys()):
#             key_date = k.split('end_fit_')[1].split('_lag')[0]
#             if key_date == fill_d:
#                 start_range = summary[k][0].index(key_date)
#                 series_fill[key_date + '_{}'.format(ix)] = summary[k][1][start_range:]
#         if len(series_fill) >=2:
#             non_zero_val_0 = np.sum(list(series_fill.values())[0]>0)
#             non_zero_val_1 = np.sum(list(series_fill.values())[1]>0)
#             non_zero = min(non_zero_val_0,non_zero_val_1)
#             plt.fill_between(actual_max_range[start_range:start_range+non_zero], list(series_fill.values())[0],
#                                     list(series_fill.values())[1], alpha = 0.2)
            
def fill_from_date(summary, fill_from_this_date, actual_max_range, min_start_plot):
            
    series_fill = {}
    start_range = (datetime.datetime.strptime(fill_from_this_date, "%Y-%m-%d")-min_start_plot).days
    for ix, k in enumerate(summary.keys()):
        key_date = k.split('end_fit_')[1].split('_lag')[0]
        if key_date == fill_from_this_date:
            select_range = summary[k][0].index(key_date)
            series_fill[key_date + '_{}'.format(ix)] = summary[k][1][select_range:]
    if len(series_fill) >=2:
        non_zero_val_0 = np.sum(list(series_fill.values())[0]>0)
        non_zero_val_1 = np.sum(list(series_fill.values())[1]>0)
        non_zero_extension = min(non_zero_val_0,non_zero_val_1)
    else:
        non_zero_extension = 0
        start_range = 0

    return series_fill, start_range, non_zero_extension
            
    


def all_data_join(dates_plot, all_data_df,analysis, summary):
    
#     all_data_df_join = pd.DataFrame(dates_plot[:fitted_day+n_prevision], columns = ['DATA'])
    all_data_df_join = pd.DataFrame(dates_plot, columns = ['DATA'])
    all_data_df_join['DATA'] = pd.to_datetime(all_data_df_join['DATA'])
    all_data_df_join.set_index('DATA', inplace=True)
    all_data_df_join = all_data_df_join.join(all_data_df, 'DATA')
    for summ in summary.items():
        summ_df = pd.DataFrame(summ[1][0], columns = ['DATA'])
        summ_df['DATA']= pd.to_datetime(summ_df['DATA'])
        summ_df.set_index('DATA', inplace=True)
        summ_df[summ[0]] = [int(s) for s in summ[1][1]]
        all_data_df_join = all_data_df_join.join(summ_df,'DATA')

    mean = all_data_df_join.drop(analysis, axis = 1).mean(1)
    mean.fillna(0, inplace=True)
    mean = mean.values

    try:
        mean = [int(x) for x in mean]
    except:
        mean = [x for x in mean]
        
    all_data_df_join['mean'] = mean
    differences = all_data_df_join.drop(analysis, axis = 1).apply(lambda x: [np.abs(x[i+1]-x[i]) \
                                                        for i in range(len(x)-1)][0] ,axis = 1)
    try:
        perc_diff = np.round((np.abs(all_data_df_join['mean']-all_data_df_join[analysis])/all_data_df_join[analysis])*100,2)
        all_data_df_join['perc_diff'] = perc_diff
    except:
        pass
    error = [np.ceil(x/2) for x in differences]

    all_data_df_join['error'] = error
    
    return all_data_df_join


def saver(path_fig, fig, exclude_test, analysis_id, n_prevision
           , path_csv, all_data_df_join, path_latex, mapper):
    
    if path_fig:
        if os.path.exists(path_fig):
            if exclude_test:
                plt.savefig(path_fig + '{}_prevision_{}_excluded_test.jpg'\
                            .format(analysis_id, n_prevision))
            else:
                plt.savefig(path_fig + '{}_prevision_{}.jpg'.format(analysis_id,n_prevision))
        else:
            os.makedirs(path_fig)
            if exclude_test:
                plt.savefig(path_fig + '{}_prevision_{}_excluded_test.jpg'\
                            .format(analysis_id,n_prevision))
            else:
                plt.savefig(path_fig + '{}_prevision_{}.jpg'.format(analysis_id,n_prevision))
                
    if path_csv:         
        if os.path.exists(path_csv):
            if exclude_test:
                all_data_df_join.to_csv(path_csv + 'report_{}_prevision_{}_excluded_test.csv'\
                                   .format(analysis_id,n_prevision))
            else:
                all_data_df_join.to_csv(path_csv + 'report_{}_prevision_{}.csv'\
                                   .format(analysis_id,n_prevision))
        else:
            os.makedirs(path_csv)
            if exclude_test:
                all_data_df_join.to_csv(path_csv + 'report_{}_prevision_{}.csv'\
                                   .format(analysis_id,n_prevision))
            else:
                all_data_df_join.to_csv(path_csv + 'report_{}_prevision_{}.csv'\
                                   .format(analysis_id,n_prevision))
    if path_latex:
        if os.path.exists(path_latex):
            if  exclude_test:
                latex_report(all_data_df_join, path_latex, name = analysis_id + '_excluded_test',
                             prevision= n_prevision, dic = mapper)
            else:
                latex_report(all_data_df_join, path_latex, name = analysis_id,
                             prevision= n_prevision, dic = mapper)
        else:
            os.makedirs(path_latex)
            if exclude_test:
                latex_report(all_data_df_join, path_latex, name = analysis_id + '_excluded_test',
                             prevision= n_prevision, dic = mapper)
            else:
                latex_report(all_data_df_join, path_latex, name = analysis_id,
                             prevision= n_prevision, dic = mapper)