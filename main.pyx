import numpy as np
from sklearn import datasets

map_size=[20,20]
learning_rate=0.1
iterations=2000


#compute the neighboorhood value
#we only want to update the close neurons
cdef neighborhood_function(s,r,to=1):
    return np.exp(-np.linalg.norm(np.array(s)-np.array(r))/2*to)

#compute the distance between one sample of the dataset X and the neurons of the som
#return the place of the closest neuron's SOM
def min_distance(xi,som):
    tmp=np.where(np.linalg.norm(xi-som,axis=-1)==np.min(np.linalg.norm(xi-som,axis=-1)))
    min_x=tmp[0][0]
    min_y=tmp[1][0]
    return [min_x,min_y]

#update the neurons to be closer of the data sample
def apply_update(som,place,xi): 
    cdef int i=0,j=0;
    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            #compute the distance between one sample of X (xi) and the neuron[i,j]
            distance=xi-som[i,j]
            #update this neuron in order to be closer of xi
            som[i,j]+=learning_rate*neighborhood_function(place,[i,j],1)*distance

def main(X,som,int iterations):
    cdef int i=0;
    cdef int j=0;
    #for each iteration, we get through all the dataset
    #and update the SOM neuron's for each sample
    for i in range(iterations):
        for j in range(X.shape[0]):
            place=min_distance(X[j],som)
            apply_update(som,place,X[j])






