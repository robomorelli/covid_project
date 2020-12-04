import numpy as np

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

dic = {'linear':{'func_fit':lin_func_fit}}

def _forbidden_fruit():
    print('apple')
    