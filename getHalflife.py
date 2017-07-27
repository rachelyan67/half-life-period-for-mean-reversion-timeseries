import numpy as np
import statsmodels.api as sm

def getHalflife(z):
    
    prez = z[:-1].tolist()
    prez.insert(0,z[0])
    prez = np.array(prez)
    dz = z - prez
    dz = dz[1:]
    prez = prez[1:]
    # assumes dz=theta*(z-mean(z))dt+w,
    # where w is error term
    model = sm.OLS(dz,prez-np.mean(prez))
    theta = model.fit().params[0]
    halflife = -np.log(2)/theta
    
    return halflife
