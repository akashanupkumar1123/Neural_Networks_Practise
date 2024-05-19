Neural Networks for Handwritten Digit Recognition, Binary



Tensorflow and Keras
Tensorflow is a machine learning package developed by Google. In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0. Keras is a framework developed independently by François Chollet that creates a simple, layer-centric interface to Tensorflow. This course will be using the Keras interface.


2 - Neural Networks
In Course 1, you implemented logistic regression. This was extended to handle non-linear boundaries using polynomial regression. For even more complex scenarios such as image recognition, neural networks are preferred.


2.1 Problem Statement
In this exercise, you will use a neural network to recognize two handwritten digits, zero and one. This is a binary classification task. Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. You will extend this network to recognize all 10 digits (0-9) in a future assignment.

This exercise will show you how the methods you have learned can be used for this classification task.


2.2 Dataset
You will start by loading the dataset for this task.

The load_data() function shown below loads the data into variables X and y
The data set contains 1000 training examples of handwritten digits 1, here limited to zero and one.

Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
Each training example becomes a single row in our data matrix X.
This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image.
𝑋=−−−(𝑥(1))−−−−−−(𝑥(2))−−−⋮−−−(𝑥(𝑚))−−−
The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.




View the variables
Let's get more familiar with your dataset.

A good place to start is to print out each variable and see what it contains.
The code below prints elements of the variables X and y.



 Check the dimensions of your variables
Another way to get familiar with your data is to view its dimensions. Please print the shape of X and y and see how many training examples you have in your dataset.


Visualizing the Data
You will begin by visualizing a subset of the training set.

In the cell below, the code randomly selects 64 rows from X, maps each row back to a 20 pixel by 20 pixel grayscale image and displays the images together.
The label for each image is displayed above the image



 Model representation
The neural network you will use in this assignment is shown in the figure below.

This has three dense layers with sigmoid activations.
Recall that our inputs are pixel values of digit images.
Since the images are of size  20×20 , this gives us  400  inputs



The parameters have dimensions that are sized for a neural network with  25  units in layer 1,  15  units in layer 2 and  1  output unit in layer 3.

Recall that the dimensions of these parameters are determined as follows:

If network has  𝑠𝑖𝑛  units in a layer and  𝑠𝑜𝑢𝑡  units in the next layer, then
𝑊  will be of dimension  𝑠𝑖𝑛×𝑠𝑜𝑢𝑡 .
𝑏  will a vector with  𝑠𝑜𝑢𝑡  elements
Therefore, the shapes of W, and b, are

layer1: The shape of W1 is (400, 25) and the shape of b1 is (25,)
layer2: The shape of W2 is (25, 15) and the shape of b2 is: (15,)
layer3: The shape of W3 is (15, 1) and the shape of b3 is: (1,)
Note: The bias vector b could be represented as a 1-D (n,) or 2-D (n,1) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention.


Tensorflow Model Implementation
Tensorflow models are built layer by layer. A layer's input dimensions (𝑠𝑖𝑛 above) are calculated for you. You specify a layer's output dimensions and this determines the next layer's input dimension. The input dimension of the first layer is derived from the size of the input data specified in the model.fit statment below.

Note: It is also possible to add an input layer that specifies the input dimension of the first layer. For example:
tf.keras.Input(shape=(400,)),    #specify input shape
We will include that here to illuminate some model sizing.



The output of the model is interpreted as a probability. In the first example above, the input is a zero. The model predicts the probability that the input is a one is nearly zero. In the second example, the input is a one. The model predicts the probability that the input is a one is nearly one. As in the case of logistic regression, the probability is compared to a threshold to make a final prediction.

NumPy Model Implementation (Forward Prop in NumPy)
As described in lecture, it is possible to build your own dense layer using NumPy. This can then be utilized to build a multi-layer neural network.


Vectorized NumPy Model Implementation (Optional)
The optional lectures described vector and matrix operations that can be used to speed the calculations. Below describes a layer operation that computes the output for all units in a layer on a given input example:

The full operation is  𝐙=𝐗𝐖+𝐛 . This will utilize NumPy broadcasting to expand  𝐛  to  𝑚  rows.



In the last example,  𝐙=𝐗𝐖+𝐛  utilized NumPy broadcasting to expand the vector  𝐛 . If you are not familiar with NumPy Broadcasting, this short tutorial is provided.

𝐗𝐖  is a matrix-matrix operation with dimensions  (𝑚,𝑗1)(𝑗1,𝑗2)  which results in a matrix with dimension  (𝑚,𝑗2) . To that, we add a vector  𝐛  with dimension  (1,𝑗2) .  𝐛  must be expanded to be a  (𝑚,𝑗2)  matrix for this element-wise operation to make sense. This expansion is accomplished for you by NumPy broadcasting.

Broadcasting applies to element-wise operations.
Its basic operation is to 'stretch' a smaller dimension by replicating elements to match a larger dimension.

More specifically: When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimensions and works its way left. Two dimensions are compatible when

they are equal, or
one of them is 1
If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the size that is not 1 along each axis of the inputs.

Here are some examples:

missing
Calculating Broadcast Result shape
The graphic below describes expanding dimensions. Note the red text below:

missing
Broadcast notionally expands arguments to match for element wise operations


The graphic above shows NumPy expanding the arguments to match before the final operation. Note that this is a notional description. The actual mechanics of NumPy operation choose the most efficient implementation.





