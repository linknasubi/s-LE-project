import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
import time
from mpl_toolkits.mplot3d import Axes3D



def Gauss_Kern(Seq:np.array, Seq_2:np.array) -> (list, list):


    
    K = ((np.linalg.norm(Seq)**2) + (np.linalg.norm(Seq_2)**2) - (2*np.inner(Seq, Seq_2)))
    
    if (np.var(distance.euclidean(Seq, Seq_2))) == 0:
        
        K = K 
    
    else:
        K = K / (2*np.var(distance.euclidean(Seq, Seq_2)))


    K = np.exp(-K)
##    
#    K = K / (1.0 + K.sum()) 
    return K


#test = Gauss_Kern(data_test2.loc[0][:-1], data_test2.loc[1][:-1])

def Adap_Neigh(dataset:pd.DataFrame) -> np.array:
    
    print("Starting Adaptive Neighborhod")
    
    AS_Avr = []
    Nw_aux = []
    Nb_aux = []
    Nw = []
    Nb = []
    pairwise_avrgs = []
    
    '''
        Average of Similarity
    
    '''
    
    for i in range(len(dataset['Class'])):

        
        for j in range(len(dataset['Class'])):
            
            if i == j:
                
                continue
            
            else:


                pairwise_avrgs.append(Gauss_Kern(dataset.iloc[i][:-1], dataset.iloc[j][:-1]))
                
        AS_Avr.append(np.mean(pairwise_avrgs))
        pairwise_avrgs = []

    '''
        
        Comparison of Similarity
    
    '''    

    
    for i in range(len(dataset['Class'])):
        
        for j in range(len(dataset['Class'])):
            
            if i == j:
                
                continue
            
            
            else:
            
                    
                pairwise_dists = pairwise_dists = Gauss_Kern(dataset.iloc[i][:-1], dataset.iloc[j][:-1])


                
                if dataset.iloc[i]['Class'] == dataset.iloc[j]['Class'] and pairwise_dists > AS_Avr[i]:
                    
                    Nw_aux.append(np.array(dataset.iloc[j]))
                
                if dataset.iloc[i]['Class'] != dataset.iloc[j]['Class'] and pairwise_dists > AS_Avr[i]:
                    
                    Nb_aux.append(np.array(dataset.iloc[j]))
        
        Nw.append(Nw_aux)
        Nb.append(Nb_aux)
        Nb_aux = []
        Nw_aux = []
        
                
    print("Ending Adaptive Neighbourhood")
    
    return Nw, Nb

#print(len(data_test[0]))

def Matrx_Affnty(dataset:pd.DataFrame) -> np.array:
    
    
    '''
        
        Set of the weights
    
    '''
    print("Starting Matrx_Affnty")
    
    resultb = []
    resultw = []
    
    N = Adap_Neigh(dataset)
    Nw = np.array(N[0])
    Nb = np.array(N[1])
    dataset = np.array(dataset)
    
    Ww = np.zeros((len(dataset), len(dataset)))
    Wb = np.zeros((len(dataset), len(dataset)))
    
    Diw = np.zeros((len(dataset), len(dataset)))
    Dib = np.zeros((len(dataset), len(dataset)))
    
    for i in range(len(dataset)):

        for x in Nw[i]:
            result = int(np.where(np.all(x==dataset, axis=1))[0][0])
            Ww[i][result] = Gauss_Kern(dataset[i][:-1], dataset[result][:-1])
        
        

        for x in Nb[i]:
            result = int(np.where(np.all(x==dataset, axis=1))[0][0])
            Wb[i][result] = 1
    
    for i in range(len(Ww)):
        Diw[i][i] = np.sum(Ww[i])
        Dib[i][i] = np.sum(Wb[i])
    
    
    print("Done")
    
    return Ww, Wb, Diw, Dib, Nw, Nb



feat_data = ["MolWt", "TPSA", "LogP", "NHA", "NHD", "NAR", "NRB", "fcSP3", "f[ARG]", "f[LYS]"]

data_col = [["Unnamed: 0", "MolWt", "TPSA", "LogP", "NHA", "NHD", "f[ARG]", "f[LYS]"],
            ["Unnamed: 0", "MolWt", "TPSA", "LogP", "NHA", "NHD", "NAR", "NRB", "fcSP3", "f[ARG]"],
            ["Unnamed: 0", "LogP", "NHA", "NHD", "NAR", "NRB", "fcSP3"],
            ["Unnamed: 0", "TPSA","NHA", "NHD", "NAR", "NRB" ],
            ["Unnamed: 0", "f[ARG]", "f[LYS]", "NRB"],
            ["Unnamed: 0", "NRB", "MolWt", "TPSA"],
            ["Unnamed: 0", "f[ARG]", "f[LYS]"],
            ["Unnamed: 0", "NAR", "f[LYS]"], 
            ["Unnamed: 0" ,"NRB"],
            ["Unnamed: 0"]]


data_col = ["Unnamed: 0"]

for i in range(len(data_col)):

    data_used = pd.read_csv('Data_Set/data_test.csv').drop(columns=data_col[i])
    
    data_test = np.array(data_used)
    
    Mtrxs = Matrx_Affnty(data_used)






    Ww = Mtrxs[0]
    Wb = Mtrxs[1]
    Dw = Mtrxs[2]
    Db = Mtrxs[3]
    Nw = list(Mtrxs[4])
    Nb = list(Mtrxs[5])
    
    
    
    print(4)
    
    Lw = Dw-Ww
    Lb = Db-Wb


    eigvals, eigvecs=np.linalg.eigh(Lb+Lw)
    
    fig = plt.figure(figsize=(10,5))
    sns.scatterplot(eigvecs[:,1], eigvecs[:,2], hue=data_used['Class'], s = 100)
    plt.title("FC_"+str(i+1))
    plt.savefig("FC_2D/"+"FC_"+str(i+1)+".png")
    
    
    fig = plt.figure(figsize=(10,5))
    ax = Axes3D(fig)
    ax.scatter(eigvecs[:,1], eigvecs[:,2],  eigvecs[:,3], c=data_used['Class'], s = 100)
    plt.title("FC_"+str(i+1))
    plt.savefig("FC_3D/"+"FC_"+str(i+1)+".png")


maxi = np.argmin(np.trace(np.dot(np.dot(data_test, data_test.T), Lw)))

maxi = np.dot(np.dot(data_test, data_test.T), Lb)








