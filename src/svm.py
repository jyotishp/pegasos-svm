#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
from mnist import MNIST

from pegasos import *

class Dataset():

    def __init__(self, data_dir, labels_to_load=[0,1]):
        self.labels_to_load = labels_to_load
        self.mnist_loader = MNIST(data_dir)
        print('Loading dataset...')

        self.xtrain, self.ytrain = self.mnist_loader.load_training()
        self.xtrain = np.array(self.xtrain, dtype=np.float64)
        self.ytrain = np.array(self.ytrain, dtype=np.float64)
        self.xtrain, self.ytrain = self.trim_dataset(self.xtrain, self.ytrain)

        self.xtest, self.ytest = self.mnist_loader.load_testing()
        self.xtest = np.array(self.xtest, dtype=np.float64)
        self.ytest = np.array(self.ytest, dtype=np.float64)
        self.xtest, self.ytest = self.trim_dataset(self.xtest, self.ytest)
        print('Dataset loaded')

    def trim_dataset(self, x, y):
        xtrain = []
        ytrain = []
        for i in range(len(y)):
            if y[i] == 0:
                ytrain.append(-1)
                xtrain.append(x[i])
            elif y[i] == 1:
                ytrain.append(1)
                xtrain.append(x[i])
            else:
                pass
        return np.array(xtrain), np.array(ytrain)

def kernel_function(x, y):
    mean = np.linalg.norm(x - y)**2
    variance = 1
    return np.exp(-mean/(2*variance))

def parse_arguments():
    # args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--kernel', default=False, action='store_true')
    parser.add_argument('--lambda', default=1, type=float)
    return parser.parse_args()

def kernelized_svm(args, data):
    weights = kernelized_pegasos(
            x=data.xtrain,
            y=data.ytrain,
            kernel=kernel_function,
            iterations=args.iterations
    )
    errors = 0
    for i in range(len(data.ytest[:500])):
        decision = 0
        for j in range(len(data.ytrain)):
            decision += weights[j]*data.ytrain[j]*kernel_function(data.xtrain[j], data.xtest[i])
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != data.ytest[i]: errors += 1
    return 1 - errors/len(data.ytest)

def svm(args, data):
    weights = pegasos(
            x=data.xtrain,
            y=data.ytrain,
            iterations=args.iterations
    )
    errors = 0
    for i in range(len(data.ytest)):
        decision = weights @ data.xtest[i].T
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != data.ytest[i]: errors += 1
    return 1 - errors/len(data.ytest)

def main():
    args = parse_arguments()
    data = Dataset(args.dataset_dir)

    if args.kernel:
        print('Using RBF kernel')
        accuracy = kernelized_svm(args, data)
    else:
        accuracy = svm(args, data)
    print('Accuracy:', accuracy)

main()
