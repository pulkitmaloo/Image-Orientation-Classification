# Image Orientation classification

## Data Description

- `photo_id`: a photo ID for the image
- `correct_orientation`: 0, 90, 180, 270 which is the orientation label
- features: 192 dimensional feature vectors, where each feature vector represent either one of the red, green, blue pixel value in range 0-255
- `train-data.txt`: 36,976 training images
- `test-data.txt`: 943 test images

```
[maloop@silo hankjang-maloop-shynaras-a4]$ wc -l train-data.txt test-data.txt
   36976 train-data.txt
     943 test-data.txt
   37919 total
```

## Classifier1: `nearest`

1. Description of how we formulated the problem including precisely defining the abstractions

k-nearest neighbor algorithm simply finds k nearest samples of a specific object, then assigns a label to the object based on the majority vote. In short, labels of the k nearest neighbors determines the label of the object.
    
There were some issues when using k-nearest neighbor. If small k was used, the classification was vulnerable to noises. However, if large k was used, it included many points from other classes. Hence, we experimented with many different values of `k`. Details are explained in part 3.    

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `nearest_model.txt`

```
./orient.py train train-data.txt nearest_model.txt nearest
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `nearest_model.txt`.

```
./orient.py test test-data.txt nearest_model.txt nearest
```

3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

As discussed above, If small k was used, the classification was vulnerable to noises. However, if large k was used, it included many points from other classes. Hence, we experimented with different k's in (3, 4, ..., sqrt(n)). The accuracy did not change much for different values of k, it was always between 68-71%, which tells us something about the dataset that there may be clusters in the data. The best test accuracy was 71%, achieved with k=9.
```
('Accuracy', 71.0, '%')
('Time taken', 100, 'seconds')
```

We also tried to reduce the dimension of the data to see whether accuracy goes up or not. We tried many different dimension of reducing the dataset, and found that using around 30 different eigenvectors worked fairly well. As you can see from the result below, the accuracy was about the same, but since the dimension was reduced a lot (previous 192 features, now 30 eigenvectors) the time needed for classification reduced even if you include the procedure of PCA. This is because kNN loops over the whole data points to find the k nearest neighboring points.
```
('Accuracy', 71.0, '%')
('Time taken', 32, 'seconds')
```

One interesting thing we've noticed was that using only blue pixels alone gave a similar result with using all features together. Here's a result of using only the blue pixels for training the model and testing using only the blue pixels. We got the following result from using k=5, usign only the blue pixels.
```
('Accuracy', 71.0, '%')
('Time taken', 40, 'seconds')
```

We've also tried another dimensional reduction algorithm Non-negative Matrix Factorization ("NMF") for experiments. Other than using additional algorithms, we've tried to apply PCA or NMF for each of the color pixels. In other words, we implemented three PCA procedures, one per each color pixel when processing the data. However, we could not get the test accuracy to get over 71%.

## Classifier2: `adaboost`

1. Description of how we formulated the problem including precisely defining the abstractions

AdaBoost algorithm is an ensemble method that utilizes a base learning algorithm, then generate many other weak learners to later be used in majority voting for the final classification. AdaBoost is simple, has solid theoretical foundation and performs well when tested in many different domains.
    
AdaBoost first assigns equal weights to training data. AdaBoost calls the base learning algorithm to the data set and the distribution of the weights, then generate a base (weak) learner `h`. After being tested by training examples, the weights get updated; if there are incorrectly classified examples, the weights would increase. From these, another weak learner is generated. This procedure is repeated for `T` times, and the final classification is done by majority vote from `T` different learners.

In this problem, we used simple decision stumps that compares one entry in teh image matrix to another. There were 192 possible combinations to generate random pairs to try. Details are explained in part 3.

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `adaboost_model.txt`

```
./orient.py train train-data.txt adaboost_model.txt adaboost
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `adaboost_model.txt`.

```
./orient.py test test-data.txt adaboost_model.txt adaboost
```

3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

In this problem, we had to deal with a multi-class problem. In order to accomplish this, we decomposed the multi-class task to a series of binary tasks. Which means is that, we did series of one vs one classification such as (0 vs 90), (0 vs 180), (0 vs 270), (90 vs 180), (90 vs 270), and (180 vs 270). Now we have 6 sets of training data.

For each training set, we take a majority vote. Let x represent a row. Let x[4] represent 4th variable in row x and x[8] represent 8th variable in row x. If x[4] - x[8] >=0, then we label a new variable as "Positive". Otherwise, we labeled that as "Negative". For instance, out of 700 rows (based on new variable in "Positive"), lets say, 650 of them were 0 degrees. In this case we have 650 rights, and 50 wrongs. Then we assigned "Positive Class" as 0 degrees. 

This works similarly with labels with "Negative". For instance, out of 900 rows (based on new variable in "Negative"), 800 of them were 90 degrees. Hence, we got 800 rights, and 100 wrongs. In the previous cases, total rights are 650 + 800 = 1450 and total wrongs are 50 + 100 = 150. From these, initial error would be 150/1600.

The weights are initialized as 1/N. As explained above, whenever we get the correct answer, we decresed weights, and otherwise increase the weights. We then normalize the weights, so the weights of misclassified items increase. Then we move to the next decision stump.

In this way of implementation, we got around 66 to 70 percent accuracy.


## Classifier3: `nnet`

1. Description of how we formulated the problem including precisely defining the abstractions

The neural network is very complex and can be configured with many different features. We could experiment with many different configurations of the architecture of the network as well as add many more features to the network to make run faster as well as converge smoothly to generalize to the dataset. 

Following are the features of our neural network:
1. We implemented the neural network in such a way as to make it easy to experiment with different architectures of the network. Our neural network can change its architecture entirely according to the parameters passed to the network
2. We used the `cross entropy cost` to calculate the error
3. We implemented `batch gradient descent` instead of stochastic gradient descent which works way faster and converges smoothly to a minima
4. We used `He initialization` for the weights so as to start at a better spot and converge faster
5. To avoid overfitting the training dataset, we added the following features to our network:
   * We implemented `dropout`
   * We implemented `L2 regularization`
6. The neural network can easily change its activation function as well to relu, softmax, sigmoid, tanh according to the parameters passed. By default, we have set the activation layers to `relu` for all the hidden layers and `softmax` at the last layer which worked best for this dataset
7. We implemented `alpha decay` so as to converge smoothly

After implementing the network we experimented a lot with the features that we added in order to increase our accuracy. We've experimented the classifer by varying alpha, number of hidden layers, number of neurons, lambda, dropout probability, and activation function.  Details are described in section 3.

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `nnet_model.txt`

```
./orient.py train train-data.txt nnet_model.txt nnet
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `nnet_model.txt`.

```
./orient.py test test-data.txt nnet_model.txt nnet
```

3. Discussion of any problems, assumptions, simplification, and/or design decisions made

Some observations we found while experimenting with different configurations of the network:

* We noticed that relu on all hidden layers and softmax on the last layer worked best. 

* He initiliazation reduced the initial cross entropy cost by half. Thus, converging faster in lesser number of iterations.

* we could see that alpha played a big role in the performance of the Neural Network. Hence, we decided to let alpha start with some big number, then decreased it as iterations increased. In this way, we thought the algorithm would prevent getting stuck in a local minimum (bigger alpha), and later converge into a reasonable minimum point (smaller alpha in the end). 

* Since we implemented regularization techniques like dropout and L2 regularization, we noticed that mkaing the architecture of the network more complex was not overfitting the datset but improving our test accuracy. 

* We had one problem: as the iteration increase, Neural Network overfitted to the training data after a certain point. Hence, we had to carefully choose the starting alpha, and experiment with diverse ratio of alpha to be decreased to prevent Neural Network to be overfitted to the training set. Surprisingly, using only one hidden layer with 193 neurons in that layer worked fairly well. Following is the parameter set we found after many experiments with parameters, hidden layers, and number of neurons. Following result is from one hidden layer (193 neurons)

```
('Test', '75.0%', 'train', '79.0%', 'cross entropy', 0.54874530470275185, 'alpha', 0.5, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 193, 4], 'PCA', False)
```

Here's another experiment that gave us 75% accuracy on the test set. Following result is from four hidden layers (8, 6, 7, 5) neurons.
```
('Test', '75.0%', 'train', '77.0%', 'cross entropy', 0.58189618939350252, 'alpha', 0.02, 'iterations', 10000, 'lambd', 0.05, 'Time', 1240, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
```

We've tried initializing He, and even implemented dropout. For an experiment with 4 hidden layers using dropout, the training accuracy actually increased (not overfitting), and test accuracy increased slightly to around 76%. Here's the final result.
```
('Test', '76.0%', 'train', '79.0%', 'cross entropy', 0.56767528801623368, 'alpha', 0.6, 'iterations', 2000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 3551, 'layers', [192, 193, 64, 4], 'PCA', False)
```

Other Experiments:
```
('Test', '73.0%', 'train', '76.0%', 'cross entropy', 0.5914779638070724, 'alpha', 0.3, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 16, 4], 'PCA', False)
('Test', '74.0%', 'train', '78.0%', 'cross entropy', 0.56314676841124767, 'alpha', 0.3, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 193, 4], 'PCA', False)
('Test', '75.0%', 'train', '79.0%', 'cross entropy', 0.54874530470275185, 'alpha', 0.5, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 193, 4], 'PCA', False)
('Test', '74.0%', 'train', '78.0%', 'cross entropy', 0.53599437926717441, 'alpha', 0.5, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 193, 130, 4], 'PCA', False)
('Test', '74.0%', 'train', '78.0%', 'cross entropy', 0.54824290828604172, 'alpha', 0.5, 'iterations', 2000, 'lambd', 0, 'layers', [192, 193, 130, 4], 'PCA', False)
('Test', '73.0%', 'train', '78.0%', 'cross entropy', 0.55188726753853878, 'alpha', 0.5, 'iterations', 5000, 'lambd', 0, 'layers', [192, 130, 4], 'PCA', False)
('Test', '74.0%', 'train', '78.0%', 'cross entropy', 0.56243754263001733, 'alpha', 0.125, 'iterations', 10000, 'lambd', 0.1, 'layers', [192, 16, 4], 'PCA', False)
('Test', '74.0%', 'train', '77.0%', 'cross entropy', 0.57976600236243447, 'alpha', 0.125, 'iterations', 20000, 'lambd', 0.1, 'layers', [192, 16, 4], 'PCA', False)
('Test', '74.0%', 'train', '77.0%', 'cross entropy', 0.57965485907628056, 'alpha', 0.125, 'iterations', 20000, 'lambd', 0.01, 'layers', [192, 16, 4], 'PCA', False)
('Test', '74.0%', 'train', '78.0%', 'cross entropy', 0.52366636304239556, 'alpha', 0.2, 'iterations', 20000, 'lambd', 0.1, 'layers', [192, 16, 16, 4], 'PCA', False)
('Test', '73.0%', 'train', '75.0%', 'cross entropy', 0.63716698428350682, 'alpha', 0.2, 'iterations', 2000, 'lambd', 0.1, 'Time', 185, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
('Test', '72.0%', 'train', '74.0%', 'cross entropy', 0.75689462861694456, 'alpha', 0.02, 'iterations', 2000, 'lambd', 0.1, 'Time', 190, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
('Test', '72.0%', 'train', '74.0%', 'cross entropy', 0.75689462861694456, 'alpha', 0.02, 'iterations', 2000, 'lambd', 0.1, 'Time', 186, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
('Test', '75.0%', 'train', '77.0%', 'cross entropy', 0.58189618939350252, 'alpha', 0.02, 'iterations', 10000, 'lambd', 0.05, 'Time', 1240, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
('Test', '72.0%', 'train', '75.0%', 'cross entropy', 0.6128374065182165, 'alpha', 0.02, 'iterations', 10000, 'lambd', 0.05, 'Time', 942, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
('Test', '73.0%', 'train', '78.0%', 'cross entropy', 0.53774609849352117, 'alpha', 1, 'iterations', 10000, 'lambd', 0.1, 'Time', 4654, 'layers', [192, 193, 16, 4], 'PCA', False)
('Test', '72.0%', 'train', '74.0%', 'cross entropy', 0.68107515367792371, 'alpha', 0.5, 'iterations', 1000, 'lambd', 0.1, 'Time', 53, 'layers', [64, 16, 4], 'PCA', False)
('Test', '72.0%', 'train', '74.0%', 'cross entropy', 0.7011610388112991, 'alpha', 0.3, 'iterations', 1000, 'lambd', 0.1, 'Time', 45, 'layers', [64, 16, 4], 'PCA', False)
('Test', '72.0%', 'train', '76.0%', 'cross entropy', 0.63815855109221942, 'alpha', 0.3, 'iterations', 1000, 'lambd', 0.1, 'Time', 83, 'layers', [192, 16, 4], 'PCA', False)
('Test', '73.0%', 'train', '78.0%', 'cross entropy', 0.53794783161106918, 'alpha', 1, 'iterations', 10000, 'lambd', 0.1, 'Time', 12436, 'layers', [192, 193, 16, 4], 'PCA', False)
('Test', '76.0%', 'train', '79.0%', 'cross entropy', 0.52401208775124242, 'alpha', 0.3, 'iterations', 3000, 'lambd', 0.7, 'Time', 1035, 'layers', [192, 128, 64, 4], 'PCA', False)
('Test', '76.0%', 'train', '79.0%', 'cross entropy', 0.51418333469315292, 'alpha', 0.3, 'iterations', 4000, 'lambd', 0.7, 'Time', 1405, 'layers', [192, 128, 64, 4], 'PCA', False)
('Test', '76.0%', 'train', '81.0%', 'cross entropy', 0.46701730873939951, 'alpha', 0.3, 'iterations', 10000, 'lambd', 0.7, 'Time', 3484, 'layers', [192, 128, 64, 4], 'PCA', False)
('Test', '72.0%', 'train', '76.0%', 'cross entropy', 0.57763705630044049, 'alpha', 0.3, 'iterations', 1000, 'lambd', 0.7, 'Time', 456, 'layers', [192, 128, 64, 20, 4], 'PCA', False)
('Test', '75.0%', 'train', '79.0%', 'cross entropy', 0.57583222617170127, 'alpha', 0.3, 'iterations', 5000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 3221, 'layers', [192, 128, 64, 4], 'PCA', False)
('Test', '73.0%', 'train', '77.0%', 'cross entropy', 0.67285100514190721, 'alpha', 0.3, 'iterations', 1000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 1453, 'layers', [192, 193, 128, 64, 32, 4], 'PCA', False)
('Test', '75.0%', 'train', '77.0%', 'cross entropy', 0.6104756923616923, 'alpha', 0.3, 'iterations', 1000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 832, 'layers', [192, 193, 64, 4], 'PCA', False)
('Test', '75.0%', 'train', '78.0%', 'cross entropy', 0.5883992468877195, 'alpha', 0.3, 'iterations', 2000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 1710, 'layers', [192, 193, 64, 4], 'PCA', False)
('Test', '76.0%', 'train', '79.0%', 'cross entropy', 0.56767528801623368, 'alpha', 0.6, 'iterations', 2000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 3551, 'layers', [192, 193, 64, 4], 'PCA', False)
('Test', '76.0%', 'train', '80.0%', 'cross entropy', 0.55359832241071549, 'alpha', 0.6, 'iterations', 5000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 4371, 'layers', [192, 193, 64, 4], 'PCA', False)
('Test', '76.0%', 'train', '80.0%', 'cross entropy', 0.54841343495295936, 'alpha', 0.6, 'iterations', 10000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 13280, 'layers', [192, 193, 64, 4], 'PCA', False)
('Test', '75.0%', 'train', '81.0%', 'cross entropy', 0.51880606858132194, 'alpha', 0.5, 'iterations', 10000, 'lambd', 0.5, 'keep_prob', 0.6, 'Time', 8669, 'layers', [192, 193, 64, 4], 'PCA', False)
```

* Since the images were really small 8 x 8 only, no matter what configurations we tried, nothing gave us an accuracy over 76% on the test dataset. We believe there may be a certain threshold on the accuracy that neural network cannot cross given a smaller training data making it not so useful as compared to other simpler algorithms. We think that with better resolution of the images, we could've achieved a higher test accuracy with a deep neural network. However, with limited computation resources we could only experiment on the smaller images.


## `best_model.txt`

We used out Neural Network model as the `best_model.txt`
