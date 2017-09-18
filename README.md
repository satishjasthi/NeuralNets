The idea of Neural Networks(NN) forms the base of various deep learning models. Let's try to understand how does an NN work and what is the math behind it using tensorflow.

se- It is open source
- It can be used to build any deep learning model
- Using TensorFlow to code deep learning models makes your understanding about the model crystal clear.

## Getting started
- Install Python 2.7 from [here](https://tutorial.djangogirls.org/en/python_installation/)
- Install TensorFlow from [here](https://www.tensorflow.org/install/)
- Install Numpy form [here](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

## Understanding Neural Networks

![Neural Network](/imgs/nn.png)

Simple NN has three components:
- Input layer
- Hidden layer
- Output layer

- Input layer: encompasses feature values for a given example from given data in the form of a vector.
For example in case of image input data feature vector contains the pixel value as a vector as shown below.

![Features](/imgs/nn2.png)

Let's see how to read this input data in TensorFlow for NN shown above

```python
#imports!
import tensorflow as tf
import numpy as np

with tf.Graph().as_default() as graph:

    #inputs from input layer,which is just 3 values
    x = tf.convert_to_tensor(np.random.normal(size = (3,1))) #converting numpy array into a tensor

    #or

    x = tf.random_normal((3,1),name="x") #creating a tensor of shape(3,1)
```

Calm down, do not panic by seeing the code.

### tf.Graph():
In tensorflow initially, program builds a graph and then computes the values of your variable by passing numerical values through the graph.

For example, you can add two number in python like this:

```python
a = 2
b = 3
c = a + b
print c
>> 5
```
However, in tensorflow(tf)

```python
a = tf.constant(2,name = "a")
b = tf.constant(3,name = "b")
c = tf.add(a,b)
#or we can also use
c1 = a + b
print c
>>Tensor("Add:0", shape=(), dtype=int32)
```
It is beacuse tf creates a node which computes the sum of a and b and not the actual sum of a,b as shown below

![TF graph](/imgs/nn3.png)

To actually add two numbers you need to pass values of a and b to the graph created earlier like this

```python
#defining graph
with tf.Graph().as_default() as computational_graph:
  a = tf.constant(2,name = "a")
  b = tf.constant(3,name = "b")
  c = tf.add(a,b)

#passing values to graph and computing sum
with tf.Session(graph = computational_graph) as sess:
  print sess.run(c)
>> 5
```

You can know more about TF functions like tf.Session, tf.constant and others [here](https://www.tensorflow.org/api_docs/)

Now we understood how to create input layer of NN, Next, we'll see how to create the hidden layer.

- Hidden layer: performs two operations, first calculating z and activating neuron as shown below

![Neuron](/imgs/nn4.png)

Let's code this in TF


```python  
'''
    weights are of shape (3,4) because there are three features for each
    sample/example and each feature needs to connect with 4 neurons in hidden
    layer_1
    '''
    w1 = tf.convert_to_tensor(np.random.normal(size=(3,4))) #using numpy array
    #or
    w1 = tf.random_normal((3,4),name="w1")
```

Now we can calculate the values of z and activation as

```python
#hidden layer, where we get both z = wT*x and a = sigmoid(z)
#hl1 = hidden layer 1 which is calculating SUM(W.T*X),where W.T means transpose of W
    hl1 = tf.reduce_sum(tf.multiply(w1,x),name="hl1")
    b1 = tf.ones((w1.shape[1],1),name="b1") #bias term
    a1 = tf.sigmoid(hl1+b1,name="a1") #activation
```

Similarly, now we calculate the activation of two neurons in the output layer, for which, activation of the hidden layer becomes the input.


## Understanding Vectorized implimentation
Let's say we have three equations

```
y1 = x1*w11 + x2*w12 + x3*w13
y2 = x1*w21 + x2*w22 + x3*w23
y3 = x1*w31 + x2*w32 + x3*w33
y4 = x1*w41 + x2*w42 + x3*w43
```
We can calculate y1, y2, y3 all in a single calculation using vectorized implimentation as

```
y = w * x
```
where,

```
y = [y1,y2,y3,y4].T ie column vector of order (4,1)
w = [w11,w12w13;w21,w22,w23;w31,w32,w33;w41,w42,w43], is a matrix of order (4,3)
x = [x1,x2,x3].T , of order (3,1)
```
in case of w(mXn),
m=4 = Number of neurons in next layer
n=3 = Number of neurons in previous layer(here it is input layer)


```python
'''
    weights between hidden layer and output layer is of shape(4,2), since there
    are 4 inputs(ie activations from hidden layer) connecting to two outputs
    '''
    w2 = tf.random_normal((4,2),name="w2")

    #output layer
    ol = tf.reduce_sum(tf.multiply(w2,a1),name="ol")
    b2 = tf.ones((w2.shape[1],1),name="b2")
    a2 = tf.sigmoid(ol+b2,name="output")

```

Finally, pass the values to TF graph to do computation

```python
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("./graphs",graph=sess.graph)
        sess.run(a2)
        writer.close()
```

We can visualize computational graph created by TF code using tensorboard, which comes along with TF. You can just run this command in terminal

```python
tensorboard --logdir="./graphs"
```
If you are using windows and jupyter-notebook use

```python
import os
os.system("tensorboard --logdir="+"./graphs")
```

Till now we have seen how NN works for a single sample/example's features, now let's see how to extend it for multiple samples using vectorization

Consider X as shown below

![Neuron](/imgs/nn5.png)

Where each column represents a sample/example

Let's implement input layer and hidden layer in TF for multiple samples

```python
with tf.Graph().as_default() as graph:
    #hidden layer and its activation
    x = tf.random_normal((3,100),name="x")
    w1 = tf.random_normal((4,3),name="w1")
    hl1 = tf.matmul(w1,x,name="hl1")
    b1 = tf.random_normal((w1.get_shape().as_list()[0],1),name="b1") #initializing bais with random values instead of ones or zeros
    a1 = tf.sigmoid(hl1+b1,name="a1")

```
Here we have 100 samples each with 3 features. hence, x is (3,100), So the output of activation node will be (4,100),ie each row will have a neuron's activation for all samples.

We can see this via tensorboard graph as

![Hidden layer](/imgs/nn6.png)

We can clearly see the dimensions of tensors flowing between nodes.

Computational_graph for entire NN looks like this

![Computational_graph](/imgs/nn7.png)
