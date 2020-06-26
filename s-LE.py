import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D






def Gauss_Kern(Seq:np.array, Seq_2:np.array) -> (list, list):


    
    K = ((np.linalg.norm(Seq)**2) + (np.linalg.norm(Seq_2)**2) -  2*np.dot(Seq, Seq_2))

    K = K/variance


    K = np.exp(-K)
    
    return K




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






data_used = pd.read_csv('Data_Set/'+'data_test'+'.csv')
data_test = np.array(data_used)


variance = np.var(data_test[:,:-1])

def Optimal_Map(n_components:int, data:pd.DataFrame, target:pd.DataFrame, scalar:float) -> np.array:
    
    pd_data = pd.concat([data, target], axis=1)
    pd_data = data_used.rename(columns={pd_data.columns[-1]: "Class"})
    pd_data.Class = pd.factorize(pd_data.Class)[0]
    
    print(pd_data)
    
    Mtrxs = Matrx_Affnty(pd_data)
    
    print(Mtrxs)
    
    Ww = Mtrxs[0]
    Wb = Mtrxs[1]
    Dw = Mtrxs[2]
    Db = Mtrxs[3]

    Lw = Dw-Ww
    Lb = Db-Wb
 
    
    B = scalar*Lb + ((1-scalar)*Ww) 
    
    
    
    eigvals, eigvecs=np.linalg.eig(B)
    
    
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    
    idx = (-1*eigvals).argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]


    eigvecs_2 = eigvecs[:,:n_components]

    return eigvecs_2

teste = Optimal_Map(n_components = 2, data = data_used.iloc[:, :-1], target = data_used.iloc[:,-1], scalar=0.55)


target = data_used.iloc[:,-1]
data = data_used.iloc[:, :-1]




fig = plt.figure(figsize=(10,5))
sns.scatterplot(teste[:,0], teste[:,1], hue=data_used['Class'], s = 100)
plt.show()



