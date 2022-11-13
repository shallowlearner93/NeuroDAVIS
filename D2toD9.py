import numpy as np
import pandas as pd

def Convert(X):
    X9D = pd.DataFrame()
    X9D[0] = X[:,0]+X[:,1]
    X9D[1] = X[:,0]-X[:,1]
    X9D[2] = X[:,0]*X[:,1]
    X9D[3] = X[:,0]**2
    X9D[4] = X[:,1]**2
    X9D[5] = (X[:,0]**2)*X[:,1]
    X9D[6] = X[:,0]*(X[:,1]**2)
    X9D[7] = X[:,0]**3
    X9D[8] = X[:,1]**3
    return X9D