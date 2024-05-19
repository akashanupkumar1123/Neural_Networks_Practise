# Planar data classification with one hidden layer

Welcome to your week 3 programming assignment! It's time to build your first neural network, which will have one hidden layer. Now, you'll notice a big difference between this model and the one you implemented previously using logistic regression.

By the end of this assignment, you'll be able to:

- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh
- Compute the cross entropy loss
- Implement forward and backward propagation


# 2 - Load the Dataset 

Now, load the dataset you'll be working on. The following code will load a "flower" 2-class dataset into variables X and Y.

Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some blue (y=1) points. Your goal is to build a model to fit this data. In other words, we want the classifier to define regions as either red or blue.


You have: - a numpy-array (matrix) X that contains your features (x1, x2) - a numpy-array (vector) Y that contains your labels (red:0, blue:1)


Simple Logistic Regression
Before building a full neural network, let's check how logistic regression performs on this problem. You can use sklearn's built-in functions for this. Run the code below to train a logistic regression classifier on the dataset.

**Interpretation**: The dataset is not linearly separable, so logistic regression doesn't perform well. Hopefully a neural network will do better. Let's try this now! 


<a name='4'></a>
## 4 - Neural Network model

Logistic regression didn't work well on the flower dataset. Next, you're going to train a Neural Network with a single hidden layer and see how that handles the same problem.

**The model**:
<img src="images/classification_kiank.png" style="width:600px;height:300px;">

**Mathematically**:

For one example $x^{(i)}$:
$$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]}\tag{1}$$ 
$$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}\tag{3}$$
$$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
$$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$

Given the predictions on all the examples, you can also compute the cost $J$ as follows: 
$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

**Reminder**: The general methodology to build a Neural Network is to:
    1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
    2. Initialize the model's parameters
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)

In practice, you'll often build helper functions to compute steps 1-3, then merge them into one function called `nn_model()`. Once you've built `nn_model()` and learned the right parameters, you can make predictions on new data.


Defining the neural network structure

Exercise 2 - layer_sizes
Define three variables: - n_x: the size of the input layer - n_h: the size of the hidden layer (set this to 4) - n_y: the size of the output layer

Hint: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

Initialize the model's parameters

Exercise 3 - initialize_parameters
Implement the function initialize_parameters().

Instructions:

Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
You will initialize the weights matrices with random values.
Use: np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b).
You will initialize the bias vectors as zeros.
Use: np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros


 forward_propagation¶
Implement forward_propagation() using the following equations:

$$Z^{[1]} = W^{[1]} X + b^{[1]}\tag{1}$$ $$A^{[1]} = \tanh(Z^{[1]})\tag{2}$$ $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}\tag{3}$$ $$\hat{Y} = A^{[2]} = \sigma(Z^{[2]})\tag{4}$$

Instructions:

Check the mathematical representation of your classifier in the figure above.
Use the function sigmoid(). It's built into (imported) this notebook.
Use the function np.tanh(). It's part of the numpy library.
Implement using these steps:
Retrieve each parameter from the dictionary "parameters" (which is the output of initialize_parameters() by using parameters[".."].
Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.


Compute the Cost
Now that you've computed A[2] (in the Python variable "A2"), which contains a[2](i) for all examples, you can compute the cost function as follows:

J=−
1
m
 
m
∑
i=1
 (y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))
 

Exercise 5 - compute_cost
Implement compute_cost() to compute the value of the cost J.

Instructions:

There are many ways to implement the cross-entropy loss. This is one way to implement one part of the equation without for loops: − 
m
∑
i=1
 y(i)log(a[2](i)):

logprobs = np.multiply(np.log(A2),Y)
cost = - np.sum(logprobs)          
Use that to build the whole expression of the cost function.

Notes:

You can use either np.multiply() and then np.sum() or directly np.dot()).
If you use np.multiply followed by np.sum the end result will be a type float, whereas if you use np.dot, the result will be a 2D numpy array.
You can use np.squeeze() to remove redundant dimensions (in the case of single float, this will be reduced to a zero-dimension array).
You can also cast the array as a type float using float().

Implement Backpropagation
Using the cache computed during forward propagation, you can now implement backward propagation.


Exercise 6 - backward_propagation
Implement the function backward_propagation().

Instructions: Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.


Figure 1: Backpropagation. Use the six equations on the right.
Tips:
To compute dZ1 you'll need to compute g[1]′(Z[1]). Since g[1](.) is the tanh activation function, if a=g[1](z) then g[1]′(z)=1−a2. So you can compute g[1]′(Z[1]) using (1 - np.power(A1, 2)).

pdate Parameters

Exercise 7 - update_parameters
Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).

General gradient descent rule: θ=θ−α
∂J
∂θ
 
 where α is the learning rate and θ represents a parameter.


Integration
Integrate your functions in nn_model()


- nn_model
Build your neural network model in nn_model().

Instructions: The neural network model has to use the previous functions in the right order.




Test the Model

5.1 - Predict

Exercise 9 - predict
Predict with your model by building predict(). Use forward propagation to predict results.

Reminder: predictions = yprediction=1{activation > 0.5}={ 
1	if activation>0.5
0	otherwise
 

As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: X_new = (X > threshold)



Test the Model on the Planar Dataset
It's time to run the model and see how it performs on a planar dataset. Run the following code to test your model with a single hidden layer of nh hidden units!


Here's a quick recap of all you just accomplished: 

- Built a complete 2-class classification neural network with a hidden layer
- Made good use of a non-linear unit
- Computed the cross entropy loss
- Implemented forward and backward propagation
- Seen the impact of varying the hidden layer size, including overfitting.

You've created a neural network that can learn patterns! Excellent work. Below, there are some optional exercises to try out some other hidden layer sizes, and other datasets




