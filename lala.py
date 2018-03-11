import matplotlib.pyplot as plt
import numpy as np

import main

map_size=[20,20]
learning_rate=0.1
iterations=10

X=np.random.randint(0, 255, (1000, 3)).astype('float64')/255

som=np.random.rand(map_size[0],map_size[1],X.shape[1])

main.main(X,som,iterations)

for i in range(map_size[0]):
    for j in range(map_size[1]):
        plt.scatter(i,j,c=som[i,j])
plt.show()