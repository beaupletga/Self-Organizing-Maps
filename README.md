# Self-Organizing-Maps

A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map, and is therefore a method to do dimensionality reduction. Self-organizing maps differ from other artificial neural networks as they apply competitive learning as opposed to error-correction learning (such as backpropagation with gradient descent), and in the sense that they use a neighborhood function to preserve the topological properties of the input space. (source : Wikipédia)

Here is the explanations of the algorithm.
![](SOM_algorithm.png)

I used Cython because of the for loop

To execute :

python lala.py

Here is an example when using SOM on colors (it's super easy to visualize) with my algorithm

![](color_figure.png)





