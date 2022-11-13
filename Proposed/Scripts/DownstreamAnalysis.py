import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, jaccard_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def correlation(dist1, dist2):
    coef, p_val = spearmanr(dist1, dist2)
    return coef

def kmeans(X,y):
    n = len(np.unique(y))
    cl = KMeans(n_clusters=n).fit_predict(X)
    
    ARI = adjusted_rand_score(cl,y)
    FMI = fowlkes_mallows_score(cl,y)
    NMI = normalized_mutual_info_score(cl,y)
    #Jaccard = jaccard_score(cl,y)
    SH_score = silhouette_score(X,cl)
    DB_score = davies_bouldin_score(X,cl)
    Score = [ARI,FMI,NMI, SH_score, DB_score]
    return Score
    
    
def Agglomerative(X,y):
    n = len(np.unique(y))
    cl = AgglomerativeClustering(n_clusters=n).fit_predict(X)
    
    ARI = adjusted_rand_score(cl,y)
    FMI = fowlkes_mallows_score(cl,y)
    NMI = normalized_mutual_info_score(cl,y)
    #Jaccard = jaccard_score(cl,y) 
    SH_score = silhouette_score(X,cl)
    DB_score = davies_bouldin_score(X,cl)
    Score = [ARI,FMI,NMI, SH_score, DB_score]
    return Score



def Knn(X,y,nei):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=nei)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    Score = [acc,pre,rec,f1]
    return Score


def RFC(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    Score = [acc,pre,rec,f1]
    return Score
    