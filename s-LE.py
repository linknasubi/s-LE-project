import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance






datapoints = pd.DataFrame(np.concatenate((np.random.rand(10,4),
                                          np.random.randint(0,2, size=(10,1))), axis=1),
                                          columns=['a', 'b', 'c', 'd','label'])



def Gauss_Kern(Seq:np.array, Seq_2:np.array) -> (list, list):

    
    
    K = ((np.linalg.norm(Seq)**2) + (np.linalg.norm(Seq_2)**2) - (2*np.inner(Seq, Seq_2)))

    K = K / (np.mean(distance.euclidean(Seq, Seq_2)))

    
    
#    K = K / np.mean(np.dot((Seq - Seq_2).T, (Seq-Seq_2))**(1/2))

    K = np.exp(-K)

    K = K / (1.0 + K.sum())
    return K

#test = Gauss_Kern(datapoints.loc[0][:-1], datapoints.loc[1][:-1])

def Adap_Neigh(dataset:pd.DataFrame) -> np.array:
    AS_Avr = []
    Nw_aux = []
    Nb_aux = []
    Nw = []
    Nb = []
    
    '''
        Average of Similarity
    
    '''
    
    for i in range(len(dataset['label'])):
        
        for j in range(len(dataset['label'])):
            
            if i == j:
                
                continue
            
            else:
                         
                pairwise_dists = Gauss_Kern(datapoints.loc[i][:-1], datapoints.loc[j][:-1])
                        
        AS_Avr.append(np.mean(pairwise_dists))
    
    
    '''
        
        Comparison of Similarity
    
    '''    
    
    
    for i in range(len(dataset['label'])):
        
        for j in range(len(dataset['label'])):
            
            if i == j:
                
                continue
            
            
            else:
            
                    
                pairwise_dists = pairwise_dists = Gauss_Kern(datapoints.loc[i][:-1], datapoints.loc[j][:-1])
                
                
                if dataset.loc[i]['label'] == dataset.loc[j]['label'] and pairwise_dists > AS_Avr[i]:
                    
                    Nw_aux.append(np.array(dataset.loc[j]))
                
                if dataset.loc[i]['label'] != dataset.loc[j]['label'] and pairwise_dists > AS_Avr[i]:
                    
                    Nb_aux.append(np.array(dataset.loc[j]))
        
        Nw.append(Nw_aux)
        Nb.append(Nb_aux)
        Nb_aux = []
        Nw_aux = []
        
                
            
    
    return Nw, Nb



def Matrc_Affnty(dataset:pd.DataFrame) -> np.array:
    
    
    '''
        
        Set of the weights
    
    '''
    
    dataset = np.array(dataset)
    Nw = np.array(Adap_Neigh(datapoints)[0])
    Nb = np.array(Adap_Neigh(datapoints)[1])
    
    Ww = np.zeros((len(dataset[0]), len(dataset[0])))
    Wb = np.zeros((len(dataset[0]), len(dataset[0])))
    
    for i in range(len(dataset[0])):
        
        for j in range(len(dataset[0])):
            
            if (any(x in dataset[j] for x in Nw[i])) and i != j:
                
                
                Ww[i][j] = Gauss_Kern(dataset[i][:-1], dataset[j][:-1])
            
            

            if (any(x in dataset[j] for x in Nb[i])) and i != j:
                
                
                Wb[i][j] = 1
                
    
    for i in range(len())
    
    
    return Ww, Wb




#def









data_test = np.array(datapoints)
Nw = Adap_Neigh(datapoints)[0]
Nb = Adap_Neigh(datapoints)[1]

Ww = Matrc_Affnty(datapoints)[0]
Wb = Matrc_Affnty(datapoints)[1]






Lw = np.identity(5, dtype=float)-Ww
Lb = np.identity(5, dtype=float)-Wb








#import networkx as nx
#
## adding a list of edges:
#G = nx.Graph()
#G.add_nodes_from([1, 2, 3, 4, 5])
#pos = nx.random_layout(G, seed=8888)
#plt.figure(figsize=[8,4])
#nx.draw(G,pos,node_size=800,font_size=16,with_labels=True)
#plt.show()
#X = np.array(list(pos.values()))
#A = pairwise_distances(X)
#plt.figure(figsize=(10,6))
#sns.set()
#sns.heatmap(A, cmap="coolwarm", annot=True);
