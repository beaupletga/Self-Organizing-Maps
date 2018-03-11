import numpy as np
from sklearn import datasets

map_size=[20,20]
learning_rate=0.1
iterations=2000

cdef neighborhood_function(s,r,to=1):
    return np.exp(-np.linalg.norm(np.array(s)-np.array(r))/2*to)


def min_distance(xi,som):
    tmp=np.where(np.linalg.norm(xi-som,axis=-1)==np.min(np.linalg.norm(xi-som,axis=-1)))
    min_x=tmp[0][0]
    min_y=tmp[1][0]
    return [min_x,min_y]


def apply_update(som,place,xi): 
    a=-som+xi
    cdef int i=0,j=0;
    # cdef float distance=0;
    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            distance=xi-som[i,j]
            som[i,j]+=learning_rate*neighborhood_function(place,[i,j],1)*distance

def main(X,som,int iterations):
    cdef int i=0;
    cdef int j=0;
    for i in range(iterations):
        for j in range(X.shape[0]):
            place=min_distance(X[j],som)
            apply_update(som,place,X[j])






