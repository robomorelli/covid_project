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
import requests

from config import *


def download_csv(zoom = 'regioni'):
    r = requests.get(source_data.replace('placeholder', zoom))
    url_content = r.content
    txt_file = open('last_update_{}.txt'.format(zoom), 'wb')
    txt_file.write(url_content)
    txt_file.close()
    read_file = pd.read_csv('last_update_{}.txt'.format(zoom))
    read_file.to_csv('last_update_{}.csv'.format(zoom))
    df = pd.read_csv('last_update_{}.csv'.format(zoom))

    os.remove('last_update_{}.txt'.format(zoom)) 
    
    return df
    

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