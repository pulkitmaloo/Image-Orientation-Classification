#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:07:53 2017

@author: pulkitmaloo
"""
from __future__ import division
from nn_test_util import test_NN


nnet = test_NN(alpha=0.3, iterations=1000, lambd=0.5, keep_prob=0.6, hidden_layers=[193, 64])