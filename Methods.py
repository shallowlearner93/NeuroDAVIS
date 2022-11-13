import numpy as np 
import pandas as pd
from sklearn.manifold import TSNE
import umap.umap_ as umap

def UseMethod(methodname, X):
    if methodname == 'TSNE':
        X_TSNE = TSNE(n_components=2).fit_transform(X)
        return X_TSNE
    
    elif methodname == 'UMAP':
        X_UMAP = umap.UMAP(n_components=2).fit_transform(X)
        return X_UMAP
    
    else:
        print('Invalid methodname')
        return None
        
        