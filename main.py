import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

map_size=[20,20]
learning_rate=0.1
iterations=1000

# X = np.array(datasets.load_iris().data)
X=np.random.randint(0, 255, (100, 3)).astype('float64')/255

# X-=np.mean(X)
# X/=np.std(X)

som=np.random.rand(map_size[0],map_size[1],X.shape[1])


def neighborhood_function(s,r,to=1):
    return np.exp(-np.linalg.norm(np.array(s)-np.array(r))/2*to)


def min_distance(xi,som):
    tmp=np.where(np.linalg.norm(xi-som,axis=-1)==np.min(np.linalg.norm(xi-som,axis=-1)))
    min_x=tmp[0][0]
    min_y=tmp[1][0]
    return [min_x,min_y]


def apply_update(som,place,xi): 
    a=-som+xi
    tmp=[[i,j] for i in range]
    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            distance=xi-som[i,j]
            som[i,j]+=learning_rate*neighborhood_function(place,[i,j])*distance

def main(X,som,iterations):
    for _ in range(iterations):
        for i in range(X.shape[0]):
            place=min_distance(X[i],som)
            apply_update(som,place,X[i])


main(X,som,iterations)

for i in range(map_size[0]):
    for j in range(map_size[1]):
        plt.scatter(i,j,c=som[i,j])
plt.show()






