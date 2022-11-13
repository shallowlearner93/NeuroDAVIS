import numpy as np
import pandas as pd

def LoadData(dataname):
    dir = 'D:/Research/NeuroDAVIS/Data/PreprocessedData/'
    dir1 = 'D:/Research/NeuroDAVIS/Data/Labels/'
    
    # -----------2D datasets-------------------
    if dataname == 'Toy':
        df=pd.read_table(dir+'Toy.txt',header=None)
        X=df.iloc[:,0:2].values
        y=df.iloc[:,2].values
        return X,y
        
    elif dataname == 'Shape':
        X=pd.read_csv(dir+'shape.csv',header=0,index_col=0)
        return np.array(X)
    
    elif dataname == 'Spiral':
        df=pd.read_table(dir+'Spiral.txt',header=None)
        X=df.iloc[:,0:2].values
        y=df.iloc[:,2].values
        return X,y
    
    elif dataname == 'R15':
        df=pd.read_table(dir+'R15.txt',header=None)
        X=df.iloc[:,0:2].values
        y=df.iloc[:,2].values
        return X,y
    
    elif dataname == '3Rings':
        X=pd.read_csv(dir+'3Rings.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'3Rings_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Olympic':
        X=pd.read_csv(dir+'Olympics.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Olympics_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'EllipticRings':
        X=pd.read_csv(dir+'EllipticRing.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'EllipticRing_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    # ---------------HD datasets-------------------
    elif dataname == 'Swiss_roll':
        X=pd.read_csv(dir+'Swiss_roll.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Swiss_roll_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Iris':
        X=pd.read_csv(dir+'Iris.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Iris_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'BreastCancer':
        X=pd.read_csv(dir+'BreastCancer.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'BreastCancer_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Wine':
        X=pd.read_csv(dir+'WIne.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Wine_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    # ---------------Text datasets-------------------
    elif dataname == 'BBCnews':
        X=pd.read_csv(dir+'BBCnews.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'BBCnews_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Spam':
        X=pd.read_csv(dir+'Spam.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Spam_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    # -----------Image datasets-------------------
    elif dataname == 'Mnist':
        X=pd.read_csv(dir+'Mnist.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Mnist_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'FMnist':
        X=pd.read_csv(dir+'FMnist.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'FMnist_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Digits':
        X=pd.read_csv(dir+'Digits.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Digits_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Coil20':
        X=pd.read_csv(dir+'Coil20.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Coil20_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Olivettifaces':
        X=pd.read_csv(dir+'Olivettifaces.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Olivettifaces_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Cifar10':
        X=pd.read_csv(dir+'Cifar10.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Cifar10_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    # -----------Single cell datasets-------------------
    elif dataname == 'PBMC3k':
        X=pd.read_csv(dir+'PBMC3k.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'PBMC3k_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Zeisel':
        X=pd.read_csv(dir+'ZeiselScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Zeisel_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Usoskin':
        X=pd.read_csv(dir+'UsoskinScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Usoskin_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Jurkat':
        X=pd.read_csv(dir+'JurkatScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Jurkat_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Kolodziejczyk':
        X=pd.read_csv(dir+'KolodziejczykScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Kolodziejczyk_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Quake':
        X=pd.read_csv(dir+'QuakeScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Quake_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    elif dataname == 'Blakeley':
        X=pd.read_csv(dir+'BlakeleyScanpy.csv',header=0,index_col=0)
        # Labels
        y=pd.read_csv(dir1+'Blakeley_groundTruth.csv',header=0,index_col=0)
        return np.array(X),np.array(y)
    
    else:
        print('Invalid dataname')
        return pd.DataFrame(), pd.DataFrame()