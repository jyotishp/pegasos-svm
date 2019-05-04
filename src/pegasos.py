#!/usr/bin/env python

import math
import numpy as np
from random import randint

def pegasos(x, y, weights=None, iterations=2000, lam=1):
    if type(weights) == type(None): weights = np.zeros(x[0].shape)
    num_S = len(y)
    for i in range(iterations):
        it = randint(0, num_S-1)
        step = 1/(lam*(i+1))
        decision = y[it] * weights @ x[it].T
        if decision < 1:
            weights = (1 - step*lam) * weights + step*y[it]*x[it]
        else:
            weights = (1 - step*lam) * weights
        #weights = min(1, (1/math.sqrt(lam))/(np.linalg.norm(weights)))*weights
    return weights

def kernelized_pegasos(x, y, kernel, weights=None, iterations=2000, lam=1):
    num_S = len(y)
    if type(weights) == type(None): weights = np.zeros(num_S)
    for _ in range(iterations):
        it = randint(0, num_S)
        decision = 0
        for j in range(num_S):
            decision += weights[j] * y[it] * kernel(x[it], x[j])
        decision *= y[it]/lam
        if decision < 1:
            weights[it] += 1
    return weights

