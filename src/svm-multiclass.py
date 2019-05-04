#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
from mnist import MNIST
import random

from pegasos import *

class Dataset():

    def __init__(self, data_dir, labels_to_load=[0,1]):
        self.labels_to_load = labels_to_load
        self.mnist_loader = MNIST(data_dir)
        print('Loading dataset...')

        self.xtrain, self.ytrain = self.mnist_loader.load_training()
        self.xtrain = np.array(self.xtrain, dtype=np.float64)
        self.ytrain = np.array(self.ytrain, dtype=np.float64)

        self.xtest, self.ytest = self.mnist_loader.load_testing()
        self.xtest = np.array(self.xtest, dtype=np.float64)
        self.ytest = np.array(self.ytest, dtype=np.float64)
        print('Dataset loaded')

    def send_data(self, id):
        dataset = {
            'data': [],
            'labels': []
        }

        for i in range(len(self.ytrain)):
            dataset['data'].append(self.xtrain[i])
            if self.ytrain[i] == id:
                dataset['labels'].append(1)
            else:
                dataset['labels'].append(-1)
        dataset['data'] = np.array(dataset['data'])
        dataset['labels'] = np.array(dataset['labels'])
        return dataset

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

def main():
    args = parse_arguments()
    data = Dataset(args.dataset_dir)
    class_weights = []
    
    for i in range(10):
        if args.kernel:
            print('Using RBF kernel')
            dataset = data.send_data(i)
            class_weights.append(kernelized_pegasos(
                x=dataset['data'],
                y=dataset['labels'],
                kernel=kernel_function,
                iterations=args.iterations
            ))
        else:
            dataset = data.send_data(i)
            class_weights.append(pegasos(
                x=dataset['data'],
                y=dataset['labels'],
                iterations=args.iterations
            ))
    
    # Testing
    errors = 0
    for i in range(len(data.ytest)):
        predictions = []
        for k in range(10):
            weights = class_weights[k]
            if args.kernel:
                decision = 0
                for j in range(len(data.ytrain)):
                    decision += weights[j]*data.ytrain[j]*kernel_function(data.xtrain[j], data.xtest[i])
            else:
                decision = weights @ data.xtest[i].T
            predictions.append(decision)
        predictions = np.array(predictions)
        class_label = predictions.argmax()
        if class_label != data.ytest[i]: errors += 1
    accuracy = 1 - errors/len(data.ytest)
    print('Error:', errors/len(data.ytest))
    print('Accuracy:', accuracy)

main()
