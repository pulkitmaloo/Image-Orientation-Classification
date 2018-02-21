#!/usr/bin/env python

### Please see README for documentation 

from __future__ import division
import sys
import timeit
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import LabelBinarizer
import cPickle
from collections import Counter
from heapq import nsmallest
import math
import random
import itertools
np.random.seed(3)

class NeuralNet(object):
    def __init__(self, alpha=0.3, iterations=4000, lambd=0.7, keep_prob=0.8, layer_dims=[192, 128, 64, 4]):
        self.set_parameters(alpha, iterations, lambd, keep_prob, layer_dims)

    def set_parameters(self, alpha, iterations, lambd, keep_prob, layer_dims):
        self.alpha = alpha
        self.iterations = iterations
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.layer_dims = layer_dims
        self.initialize_parameters(layer_dims)

    def initialize_parameters(self, layer_dims):
        self.parameters = {}
        for l in range(1, len(layer_dims)):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l - 1])
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        Z_exp = np.exp(Z - np.max(Z))
        Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
        return Z_exp / Z_sum

    def relu(self, Z):
        return np.maximum(0, Z)

    def dropout(self, A):
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < self.keep_prob
        A = A * D
        A = A / self.keep_prob
        return A, D

    def linear_fwd_activation(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b

        if activation == "sigmoid":
            A = self.sigmoid(Z)
            cache = ((A_prev, W, b), Z)

        elif activation == "relu":
            A = self.relu(Z)
            A, D = self.dropout(A)
            cache = ((A_prev, W, b), Z, D)

        elif activation == "softmax":
            A = self.softmax(Z)
            cache = ((A_prev, W, b), Z)

        return A, cache

    def forward_propogation_with_dropout(self, X):
        caches = []
        L = len(self.layer_dims) - 1

        A = X
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            A, cache = self.linear_fwd_activation(A_prev, W, b, "relu")
            caches.append(cache)

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        AL, cache = self.linear_fwd_activation(A, W, b, "softmax")
        caches.append(cache)

        return AL, caches

    def cross_entropy_softmax(self, AL, ZL, Y):
#        return - np.sum(np.multiply(Y, np.log(AL)))
        return - np.sum(np.multiply(Y, ZL - np.log(np.sum(np.exp(ZL), axis=0, keepdims=True))))

    def compute_cost(self, AL, Y, caches):
        m = Y.shape[1]
        L = len(self.layer_dims) - 1

#       For Sigmoid
#        cross_entropy_cost = (-1/m) * np.sum( np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)) )

        ZL = caches[-1][1]
        cross_entropy_cost = self.cross_entropy_softmax(AL, ZL, Y) / m

        W_sum = 0
        for l in range(1, L+1):
            W_sum += np.sum(np.square(self.parameters["W"+str(l)]))
        regularized_cost = self.lambd * W_sum / (2 * m)

        cost = np.squeeze(cross_entropy_cost) + regularized_cost
        return cost

    def linear_backward(self, dZ, cache, l):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        W = self.parameters["W" + str(l)]

        dW = (1/m) * np.dot(dZ, A_prev.T) + (self.lambd * W) / m
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def dZL(self, AL, Y):
        return AL - Y

    def linear_activation_backward(self, dA, cache, activation, l):
        ((A_prev, W, b), Z, D) = cache

        if activation == "relu":
            dA = dA * D                 # Dropout
            dA = dA / self.keep_prob
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0

        elif activation == "sigmoid":
            t = self.sigmoid(-Z)
            dZ = dA * t * (1-t)

        elif activation == "softmax":
            t = self.softmax(Z)
            dZ = t - Z

        dA_prev, dW, db = self.linear_backward(dZ, (A_prev, W, b), l)

        return dA_prev, dW, db

    def backpropogation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        cache = caches[L-1]

#        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
#        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_activation_backward(dAL, cache, "sigmoid", L)

        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_backward(self.dZL(AL, Y), cache[0], L)
        for l in range(L-1, 0, -1):
            cache = caches[l-1]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+str(l+1)], cache, "relu", l)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp

        return grads

    def lrate(self, epoch):
        drop = 0.5
        epochs_drop = 1000
        lr = self.alpha * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr

    def update_parameters(self, grads, i):
        L = len(self.layer_dims) - 1
        for l in range(1, L+1):
            self.parameters["W"+str(l)] -= self.alpha * grads["dW"+str(l)]
            self.parameters["b"+str(l)] -= self.alpha * grads["db"+str(l)]

    def train(self, X_train, y_train):
        X = X_train
        Y = y_train

        costs = [(0,0)]
        for i in xrange(self.iterations + 1):
            AL, caches = self.forward_propogation_with_dropout(X)

            cost = self.compute_cost(AL, Y, caches)

            grads = self.backpropogation(AL, Y, caches)

            self.update_parameters(grads, i)

            if i%100 == 0:
                if i % 1000 == 0:
                    self.alpha /= 1.5
                costs.append((i, cost))
                _, acc = self.test(X, Y)
                print "Iteration", i, "->", "Accuracy", acc, "|| Cost", cost

    def forward_propogation(self, X):
        L = len(self.layer_dims) - 1

        A = X
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        ZL = np.dot(W, A) + b
        AL = self.softmax(ZL)

        return AL

    def test(self, X_test, y_test):
        X_t = X_test
        Y_t = y_test
        AL = self.forward_propogation(X_t)

        original = Y_t.argmax(0)
        predicted = AL.argmax(0)
        m = len(original)
        incorrect = np.count_nonzero(original - predicted)
        return predicted, round((m - incorrect) / m, 4) * 100


class KNN(object):
    def __init__(self, k=7):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def test(self, X_test, y_test):
        correct = 0
        pred = []
        for X_ins, y_ins in zip(X_test, y_test):
            predict = self.predict(X_ins)
            pred.append(predict)
            if y_ins == predict:
                correct += 1
        return pred, round(correct/len(X_test), 4) * 100

    def predict(self, p):
        class_count = Counter(self.nearest_neighbours(p))
        return class_count.most_common()[0][0]

    def nearest_neighbours(self, p):
        distances = norm(self.X_train - p, axis=1)
        neighbors = zip(distances, self.y_train)
        k_nearest = nsmallest(self.k, neighbors, key=lambda x: x[0])
        return map(lambda x: x[1], k_nearest)

# k: number of decision stumps
class AdaBoost(object):
    def __init__(self, k = 500):
        self. k = k

    def train(self, X_train, y_train, variables):
        possible_pairs = get_possible_pairs()
        y_unique = list(set(y_train))
        vote_classifier = {}
        weights = np.array([float(1)/float(len(y_train))]*len(y_train))
        for variable in variables:
            error = 0
            index1 = variable[0]
            index2 = variable[1]
            decision_stump = []
            decision_stump_category = {'Positive':{},'Negative':{}}
            #y_category = {}
            #count_y_category = 0
            for i in xrange(0, len(X_train)):
                if X_train[i][index1] >= X_train[i][index2]:
                    decision_stump.append('Positive')
                    try:
                        decision_stump_category['Positive'][y_train[i]] += 1
                    except:
                        decision_stump_category['Positive'][y_train[i]] = 1
                        #y_category[y_train[i]] = count_y_category
                        #count_y_category += 1
                else:
                    decision_stump.append('Negative')
                    try:
                        decision_stump_category['Negative'][y_train[i]] += 1
                    except:
                        decision_stump_category['Negative'][y_train[i]] = 1
                        #y_category[y_train[i]] = count_y_category
                        #count_y_category += 1


            Positive_Class = max(decision_stump_category['Positive'], key=lambda k: decision_stump_category['Positive'][k])
            Negative_Class = max(decision_stump_category['Negative'], key=lambda k: decision_stump_category['Negative'][k])
            decision_stump_classification = []
            for i in xrange(0, len(y_train)):
                if decision_stump[i] == 'Positive':
                    decision_stump_classification.append(Positive_Class)
                else:
                    decision_stump_classification.append(Negative_Class)
                if decision_stump_classification[i] != y_train[i]:
                    error = error + weights[i]

            if error > 0.5:
                continue

            for i in xrange(0, len(y_train)):
                if decision_stump_classification[i] == y_train[i]:
                    weights[i] = weights[i]*error/(1.0-error)

            #Normalizing weights
            sum_weights = sum(weights)
            weights = weights / sum_weights
            vote_classifier[variable] = {}
            vote_classifier[variable]['weight'] = math.log((1-error) / error)
            vote_classifier[variable]['Positive_Class'] = Positive_Class
            vote_classifier[variable]['Negative_Class'] = Negative_Class

        return vote_classifier, y_unique

    def test(self, X_test, y_test, vote_classifier, y_unique):
        possible_pairs = get_possible_pairs()
        y_unique_dict = {}
        y_unique_dict[y_unique[0]] = 1
        y_unique_dict[y_unique[1]] = -1
        classification = [0] * len(y_test)
        classifiers = vote_classifier
        for classifier in classifiers:
            index1 = classifier[0]
            index2 = classifier[1]
            for i in list(range(0,len(y_test))):
                if X_test[i][index1] >= X_test[i][index2]:
                    vote = classifiers[classifier]['Positive_Class']
                    classification[i] = classification[i] + y_unique_dict[vote]*classifiers[classifier]['weight']
                else:
                    vote = classifiers[classifier]['Negative_Class']
                    classification[i] = classification[i] + y_unique_dict[vote]*classifiers[classifier]['weight']
        classification_category = []
        for i in xrange(0, len(y_test)):
            if classification[i] >= 0:
                classification_category.append(y_unique[0])
            else:
                classification_category.append(y_unique[1])
        return classification_category


def read_file(fname, shuffle_data=True):
    print "Reading data from", fname, "..."
    image = np.loadtxt(fname, usecols=0, dtype=str)  # .reshape(len(X), 1)
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)  # .reshape(len(X), 1)

    if shuffle_data:
        shuffle_indices = range(len(y))
        np.random.shuffle(shuffle_indices)
        X  = X[shuffle_indices, ]
        y = y[shuffle_indices, ]
        image = image[shuffle_indices, ]

    return list(image), X/255, y

def transform_Y_for_NN(Y):
    lb = LabelBinarizer()
    lb.fit(Y)
    return lb

def get_possible_pairs():
    global n_features
    possible_pairs = []
    for x in xrange(n_features):
        for y in xrange(x, n_features):
            if x==y:
                continue
            possible_pairs.append((x,y))
    return possible_pairs

def to_file(image, pred):
    f = open('output.txt', 'w')
    for line in xrange(len(image)):
        f.write(image[line] + ' ' + str(pred[line]) + '\n')
    f.close()


if __name__ == "__main__":
    task, fname, model_file, model = sys.argv[1:]

    image, X, y = read_file(fname, shuffle_data=True)

    if task == "train":
        print "Training", model, "model..."
        tic = timeit.default_timer()

        if model == "nearest":
            knn = KNN(k=9)
            knn.train(X, y)
            models = (knn)

        elif model == "adaboost":
            k = 500
            n_features = X.shape[1]
            possible_pairs = get_possible_pairs()
            n_labels = len(set(y))
            num_variables = int(n_labels * (n_labels-1) / 2)
            variables = []
            for i in range(num_variables):
                variables.append(random.sample(possible_pairs, k))
            y_trains = {}
            X_trains = {}
            for label in set(y):
                y_trains[label] = []
                X_trains[label] = []
            for i, label in enumerate(y):
                y_trains[label].append(y[i])
                X_trains[label].append(X[i])

            X_train_pairs = []
            y_train_pairs = []
            for c1, c2 in list(itertools.combinations(set(y),2)):
                X_train_pairs.append(X_trains[c1] + X_trains[c2])
                y_train_pairs.append(y_trains[c1] + y_trains[c2])

            adaboost = AdaBoost(k)

            vote_classifier_y_unique = []
            for i in range(num_variables):
                vote_classifier_y_unique.append(adaboost.train(X_train_pairs[i], y_train_pairs[i], variables[i]))

            models = (vote_classifier_y_unique, adaboost)

        elif model == "nnet" or model == "best":
            lb = transform_Y_for_NN(y)
            Y_lb = lb.transform(y)

            alpha = 0.3
            iterations = 2000
            lambd = 0.5
            keep_prob = 0.6
            layers = [X.shape[1]] + [193, 64] + [Y_lb.shape[1]]

            nnet = NeuralNet(alpha=alpha, iterations=iterations, lambd=lambd,
                             keep_prob=keep_prob, layer_dims=layers)

            nnet.train(X.T, Y_lb.T)

            models = (lb, nnet)
        else:
            print "Model not found"

        cPickle.dump(models, open(model_file, "wb"), protocol=2)
        toc = timeit.default_timer()
        print "Time taken", int(toc - tic), "seconds"

    else:
        print "Testing", model, "model..."
        tic = timeit.default_timer()

        models = cPickle.load(open(model_file, "rb"))

        if model == "nearest":
            knn = models
            pred, score = knn.test(X, y)
            print("Writing to a file...")
            to_file(image, pred)

        elif model == "adaboost":
            n_labels = len(set(y))
            num_variables = int(n_labels * (n_labels-1) / 2)
            n_features = X.shape[1]

            vote_classifier_y_unique, adaboost = models
            classification_category = []
            for i in range(num_variables):
                classification_category.append(adaboost.test(X, y, vote_classifier_y_unique[i][0], vote_classifier_y_unique[i][1]))

            final_classification = []
            count_correct = 0

            for i in xrange(y.shape[0]):
                lst = []
                for pair in range(num_variables):
                    lst += [classification_category[pair][i]]
                final_classification.append(max(lst,key=lst.count))
                if final_classification[i] == y[i]:
                    count_correct += 1

            score = round(float(count_correct)/float(len(y)), 2)*100

            print("Writing to a file...")
            to_file(image, final_classification)
            # f = open('output.txt', 'w')
            # for line in range(y.shape[0]):
                # f.write(image[line] + ' ' + str(final_classification[line]) + '\n')
            # f.close()

        elif model == "nnet" or model == "best":
            lb, nnet = models
            Y_lb = lb.transform(y)
            pred, score = nnet.test(X.T, Y_lb.T)
            pred *= 90
            print("Writing to a file...")
            to_file(image, pred)

        else:
            print "Model not found"

        print ("Accuracy", score, "%")
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")


