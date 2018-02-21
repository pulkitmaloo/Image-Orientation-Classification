#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 02:50:10 2017

@author: pulkitmaloo
"""
from __future__ import division

def test_NN(alpha, iterations, lambd, keep_prob, hidden_layers, REDUCE_DIM=False):

    from orient import NeuralNet, read_file, transform_Y_for_NN
    import timeit
    import numpy as np
    from random import shuffle

    np.random.seed(3)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.85, svd_solver="full")

    # Train
    X, y = read_file("train-data.txt")

#    blue = [x for x in range(X.shape[1]) if x%3==2]
#    X = X[:,blue]

    if REDUCE_DIM:
        X = pca.fit_transform(X)

    lb = transform_Y_for_NN(y)
    Y_lb = lb.transform(y)
        #nnet = NeuralNet() 64, 16

    tic = timeit.default_timer()

    layers = [X.shape[1]] + hidden_layers + [Y_lb.shape[1]]

    nnet = NeuralNet(alpha=alpha, iterations=iterations, lambd=lambd, keep_prob=keep_prob, layer_dims=layers)

    nnet.train(X.T, Y_lb.T)

    # Test on train data
    train_score = nnet.test(X.T, Y_lb.T)
    print "Train Accuracy", train_score, "%"

    # Test on test data
    X_t, y_t = read_file("test-data.txt")

#    blue = [x for x in range(X_t.shape[1]) if x%3==2]
#    X_t = X_t[:,blue]

    if REDUCE_DIM:
        X_t = pca.transform(X_t)
    Y_tlb = lb.transform(y_t)

    test_score = nnet.test(X_t.T, Y_tlb.T)
    print "Test Accuracy ", test_score, "%"

    toc = timeit.default_timer()
    time = int(toc-tic)
    print "Time", time, "seconds"

    result = ("Test", str(test_score)+"%", "train", str(train_score)+"%", "cross entropy", nnet.costs[-1][-1],
              "alpha", alpha, "iterations", iterations, "lambd", lambd, "keep_prob", keep_prob, "Time", time, "layers",
              layers, "PCA", REDUCE_DIM)

    with open("neuralnet_results.txt", "a") as fhand:
        fhand.write(str(result)+"\n")

    return nnet